import numpy as np
import matplotlib.pyplot as plt
from alfred.parameters import base_dir


from tensorflow import keras

emul = "keras_xe_emul_glx"
path = f"{base_dir}/emulators/xe_emul"

model = keras.models.load_model(f"{path}/{emul}.keras")
data = np.load(f"{path}/{emul}_pmean_pstd_zm_zs_xev.npy", allow_pickle=True)


def xe_emul_dict(
    zvect, params, He1=False, He2=False, zHe=3.5, plot=False, newfig=False
):
    """
    zvect : vect of z values at which xe should be evaluated [z increasing]
    params: dict of params values
            eg: params = {'fX':-2.71669877 ,
                          'rHS':0.2        ,
                           'tau': 3.51074603  ,
                           'Mmin':9.33       ,
                           'fesc': 0.275 }
    Emul: name of the directory containing the keras files
    allH: means H plus 1st reio of He

    RETURNS: xe values at zvect

    """

    parmeansstd = data.item()["parmeansstd"]
    zm = data.item()["zm"]
    zs = data.item()["zs"]
    xe_interp = data.item()["xe_int"]

    X0_values = np.array(
        [
            (params[key] - parmeansstd[key]["mean"]) / parmeansstd[key]["std"]
            for key in params.keys()
        ]
    )

    Yval = model.predict(X0_values[None, :], verbose=0)

    zval = (Yval.T * zs + zm).flatten()[::-1]

    x_vals = np.hstack(([1e-1, 0.98 * zval[0]], zval.flatten()))
    x_vals = np.hstack((x_vals, [1.02 * zval[-1]]))

    y_vals = np.hstack(([1, 1], xe_interp.flatten()[::-1]))
    y_vals = np.hstack((y_vals, [0.5 * y_vals[-1]]))

    xe_fin = 10 ** (
        np.interp(
            np.log10(zvect),
            np.log10(x_vals),
            np.log10(y_vals),
            left=np.log10(1),
            right=-10,
        )
    )

    if He1:
        # He1 reionization
        xe_fin = 1.08 * xe_fin

    if He2:
        # He2 reionization

        helium_fullreion_redshift = zHe  ## default 3.5
        helium_fullreion_deltaredshift = 0.5  ## default 3.5
        helium_fullreion_redshiftstart = 5.0
        yp = 0.2453
        not4 = 3.9715
        fHe = yp / (not4 * (1 - yp))

        a = 1.0 / (zvect + 1.0)

        deltayHe2 = (
            1.5
            * np.sqrt(1 + helium_fullreion_redshift)
            * helium_fullreion_deltaredshift
        )
        VarMid2 = (1.0 + helium_fullreion_redshift) ** 1.5

        xod2 = (VarMid2 - 1.0 / a**1.5) / deltayHe2

        # tgh2 = np.zeros(np.size(zvect))+1.0

        tgh2 = np.tanh(xod2)  # check if xod<100

        xe_fin = xe_fin + (fHe) * (tgh2 + 1.0) / 2.0

    if plot:
        if newfig:
            plt.figure()
        plt.plot(zvect, xe_fin)
        plt.xlabel("z")
        plt.ylabel(r"$x_e$")

    return xe_fin


def xe_emul_array(
    zvect,
    params_array,
    emul="keras_xe_emul_glx",
    He1=False,
    He2=False,
    zHe=3.5,
    plot=False,
    newfig=False,
):
    """
    zvect : vect of z values at which xe should be evaluated [z increasing]
    params: dict of params values
            eg: params_array = [[-2.77],[0.2],[3.51],[9.33],[0.275]]
    old :  params_array = [[-2.71669877] ,
                          'rHS':0.2        ,
                           'tau': 3.51074603  ,
                           'Mmin':9.33       ,
                           'fesc': 0.275 }
    Emul: name of the directory containing the keras files
    allH: means H plus 1st reio of He

    RETURNS: xe values at zvect

    """

    params_array = params_array.T  # this is to fit with the expected array shape

    if params_array.ndim == 1:
        params_array = params_array.reshape(params_array.size, 1)

    nmodels = np.shape(params_array)[1]

    keys = ["fX", "rHS", "tau", "Mmin", "fesc"]

    parmeansstd = data.item()["parmeansstd"]
    zm = data.item()["zm"]
    zs = data.item()["zs"]
    xe_interp = data.item()["xe_int"]

    X0_values = np.array(
        [
            (params_array[i, :] - parmeansstd[keys[i]]["mean"])
            / parmeansstd[keys[i]]["std"]
            for i in np.arange(params_array.shape[0])
        ]
    )

    Yval = model.predict(X0_values.T, verbose=0)

    zval = (Yval.T * zs + zm)[::-1]

    pt0 = np.zeros(nmodels) + 1e-1
    pt1 = np.zeros(nmodels) + 0.98 * zval[0, :]
    pt98 = np.zeros(nmodels) + 1.02 * zval[-1, :]
    pt99 = np.zeros(nmodels) + 2 * zval[-1, :]

    x_vals = np.vstack((pt0, pt1, zval, pt98, pt99))
    # x_vals = np.hstack((x_vals,[1.02*zval[-1]]))

    y_vals = np.hstack(([1, 1], xe_interp.flatten()[::-1]))
    y_vals = np.hstack((y_vals, [0.5 * y_vals[-1]], [1e-10]))

    xe_final = []

    for jmod in np.arange(nmodels):
        xe_fin = 10 ** (
            np.interp(
                np.log10(zvect),
                np.log10(x_vals[:, jmod]),
                np.log10(y_vals),
                left=np.log10(1),
                right=-10,
            )
        )

        if He1:
            # He1 reionization
            xe_fin = 1.08 * xe_fin

        if He2:
            # He2 reionization

            helium_fullreion_redshift = zHe  ## default 3.5
            helium_fullreion_deltaredshift = 0.5  ## default 3.5
            helium_fullreion_redshiftstart = 5.0
            yp = 0.2453
            not4 = 3.9715
            fHe = yp / (not4 * (1 - yp))

            a = 1.0 / (zvect + 1.0)

            deltayHe2 = (
                1.5
                * np.sqrt(1 + helium_fullreion_redshift)
                * helium_fullreion_deltaredshift
            )
            VarMid2 = (1.0 + helium_fullreion_redshift) ** 1.5

            xod2 = (VarMid2 - 1.0 / a**1.5) / deltayHe2

            # tgh2 = np.zeros(np.size(zvect))+1.0

            tgh2 = np.tanh(xod2)  # check if xod<100

            xe_fin = xe_fin + (fHe) * (tgh2 + 1.0) / 2.0

        xe_final.append(xe_fin)

        if plot:
            if newfig:
                plt.figure()
            plt.plot(zvect, xe_fin)
            plt.xlabel("z")
            plt.ylabel(r"$x_e$")

    xe_final = np.asarray(xe_final)

    # if 1 in xe_final.shape:
    #     return np.array(xe_final).flatten()

    return xe_final
