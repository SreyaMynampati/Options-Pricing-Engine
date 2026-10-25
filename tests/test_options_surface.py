import sys
from pathlib import Path
import csv
import json
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from volterra.options_market import CalibratedSurface, fit_surface, load_surface


class SurfaceCalibrationTests(unittest.TestCase):
    def sample(self):
        rows=[]
        ks=np.linspace(-.15,.15,9)
        for expiry, t in ((1_800_000_000_000,.25),(1_810_000_000_000,.50)):
            for k in ks:
                iv=.40-.30*k+.80*k*k
                rows.append({"expiry_ms":expiry,"time_years":t,"log_forward_moneyness":float(k),
                             "mid_iv":float(iv),"quote_status":"eligible","strike":float(np.exp(k)*100),
                             "bid_iv_exchange":float(iv-.01),"ask_iv_exchange":float(iv+.01),
                             "relative_spread":.05})
        return rows

    def test_fits_smiles_and_interpolates_total_variance(self):
        smiles, diagnostics=fit_surface(self.sample())
        self.assertEqual(len(smiles),2)
        self.assertEqual(diagnostics["calendar_total_variance_violations"],0)
        surface=CalibratedSurface(smiles)
        self.assertAlmostEqual(surface.volatility(0,.375),.4,places=10)
        self.assertAlmostEqual(surface.volatility(.1,.25),.40-.03+.008,places=10)

    def test_surface_refuses_extrapolation(self):
        surface=CalibratedSurface(fit_surface(self.sample())[0])
        with self.assertRaises(ValueError):
            surface.volatility(0,.1)
        with self.assertRaises(ValueError):
            surface.volatility(.5,.3)

    def test_saved_surface_loads(self):
        smiles,_=fit_surface(self.sample())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"calibrated_smiles.csv"
            fields=("expiry_date_utc","expiry_ms","time_years","quote_count","unique_strikes","x_scale",
                    "coefficients_iv_poly_ascending","weighted_rmse_iv","median_abs_error_iv","max_abs_error_iv",
                    "min_log_forward_moneyness","max_log_forward_moneyness","fit_method")
            with path.open("w",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                for smile in smiles:
                    row=dict(smile)
                    row["coefficients_iv_poly_ascending"]=json.dumps(row["coefficients_iv_poly_ascending"])
                    writer.writerow(row)
            loaded=load_surface(path)
            self.assertAlmostEqual(loaded.volatility(0,.375),.4,places=10)


if __name__=="__main__":
    unittest.main()
