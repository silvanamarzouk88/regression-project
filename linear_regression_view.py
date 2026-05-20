import numpy as np
from scipy import stats


class LinearRegressionStatisticalView:

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self._fitted = False

    def fit(self, X, Y):
        self.X = np.array(X, dtype=float)
        self.Y = np.array(Y, dtype=float)
        n = len(self.X)

        x_mean = np.mean(self.X)
        y_mean = np.mean(self.Y)

        Sxx = np.sum((self.X - x_mean) ** 2)
        Sxy = np.sum((self.X - x_mean) * (self.Y - y_mean))
        Syy = np.sum((self.Y - y_mean) ** 2)

        self._b1 = Sxy / Sxx
        self._b0 = y_mean - self._b1 * x_mean

        y_hat = self._b0 + self._b1 * self.X
        residuals = self.Y - y_hat

        SS_res = np.sum(residuals ** 2)
        SS_tot = Syy
        SS_mod = SS_tot - SS_res

        self._n = n
        self._df_model = 1
        self._df_resid = n - 2
        self._df_total = n - 1

        MS_mod = SS_mod / self._df_model
        MS_res = SS_res / self._df_resid

        self._f_stat = MS_mod / MS_res
        self._r2 = SS_mod / SS_tot
        self._r = np.sqrt(self._r2) * np.sign(self._b1)

        self._se2 = MS_res
        self._Sxx = Sxx
        self._x_mean = x_mean

        self._fitted = True

    def f_statistic(self) -> float:
        return float(self._f_stat)

    def f_critical(self) -> float:
        return float(stats.f.ppf(1 - self.alpha, self._df_model, self._df_resid))

    def beta_confidence_intervals(self) -> dict:
        t_crit = stats.t.ppf(1 - self.alpha / 2, self._df_resid)

        se_b1 = np.sqrt(self._se2 / self._Sxx)
        se_b0 = np.sqrt(self._se2 * (1 / self._n + self._x_mean ** 2 / self._Sxx))

        b1_lower = self._b1 - t_crit * se_b1
        b1_upper = self._b1 + t_crit * se_b1

        b0_lower = self._b0 - t_crit * se_b0
        b0_upper = self._b0 + t_crit * se_b0

        return {
            "beta_0": (float(b0_lower), float(b0_upper)),
            "beta_1": (float(b1_lower), float(b1_upper))
        }

    def degrees_of_freedom(self) -> dict:
        return {
            "df_model": self._df_model,
            "df_resid": self._df_resid,
            "df_total": self._df_total
        }

    def r_squared(self) -> float:
        return float(self._r2)

    def r(self) -> float:
        return float(self._r)

    def summary(self) -> dict:
        return {
            "f_critical": self.f_critical(),
            "f_stat": self.f_statistic(),
            "beta_confidence_intervals": self.beta_confidence_intervals(),
            "degrees_of_freedom": self.degrees_of_freedom(),
            "r_squareddef": self.r_squared(),
            "r": self.r()
        }


def compare(df, x_col_1: str, x_col_2: str, y_col: str, alpha: float = 0.05) -> dict:
    model_a = LinearRegressionStatisticalView(alpha=alpha)
    model_b = LinearRegressionStatisticalView(alpha=alpha)

    model_a.fit(df[x_col_1].values, df[y_col].values)
    model_b.fit(df[x_col_2].values, df[y_col].values)

    f1 = model_a.f_statistic()
    f2 = model_b.f_statistic()

    best = x_col_1 if f1 >= f2 else x_col_2

    return {
        "predictor_1": x_col_1,
        "predictor_2": x_col_2,
        "f_stat_1": f1,
        "f_stat_2": f2,
        "best_predictor": best
    }
