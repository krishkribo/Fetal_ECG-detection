from math import sqrt


def corr(W, m, n, l):
    result = 1
    for i in range(l - 1):
        result *= W[m + i][n]
    return result


def f(W: [float], corr2: [float], mask, m: int, n_length: int):
    PCorr = sum([corr2[n] ** 2 for n in range(n_length)])
    PW = sum([W[m][n] ** 2 for n in range(n_length)])

    corr2 = [corr2[n] * sqrt(PW / PCorr) for n in range(n_length)]

    for n in range(n_length):
        if abs(corr2[n]) != abs(W[m][n]):
            print(abs(corr2[n]), abs(W[m][n]))
        if abs(corr2[n]) > abs(W[m][n]):
            corr2[n] = 0.0
            W[m][n] = 0.0
            mask[m][n] = 1
    return corr2, PW, mask


# Spatially selective noise filtration
def ssnf(W: [[float]], scales: int, noise_thresholds: [float]):
    WW = [x[:] for x in W[:]]
    n_length = len(W[0])
    mask = []

    for m in range(scales-1):
        mask.append([0 for _ in range(n_length)])

        corr2 = [corr(W, m, n, 2) for n in range(n_length)]
        _, PW, mask = f(W=W, corr2=corr2, mask=mask, m=m, n_length=n_length)
        while PW < noise_thresholds[m]:
            print('While iter, PW=', PW)
            corr2 = [corr(W, m, n, 2) for n in range(n_length)]
            _, PW, mask = f(W=W, corr2=corr2, mask=mask, m=m, n_length=n_length)

        for n in range(n_length):
            WW[m][n] = mask[m][n] * WW[m][n]

    return WW