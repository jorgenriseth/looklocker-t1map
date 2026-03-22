use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{DVector, Dyn, OMatrix, OVector, U3};

pub struct LookLockerProblem {
    /// Time points
    pub t: DVector<f64>,
    /// Observed signal
    pub y: DVector<f64>,
    /// Parameters: x1, x2, x3
    pub p: OVector<f64, U3>,
}

impl LeastSquaresProblem<f64, Dyn, U3> for LookLockerProblem {
    type ParameterStorage = nalgebra::storage::Owned<f64, U3>;
    type ResidualStorage = nalgebra::storage::Owned<f64, Dyn>;
    type JacobianStorage = nalgebra::storage::Owned<f64, Dyn, U3>;

    fn set_params(&mut self, p: &OVector<f64, U3>) {
        self.p = *p;
    }

    fn params(&self) -> OVector<f64, U3> {
        self.p
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        let x1 = self.p[0];
        let x2 = self.p[1];
        let x3 = self.p[2];

        let residuals: Vec<f64> = self
            .t
            .iter()
            .zip(self.y.iter())
            .map(|(&t, &y)| {
                let term_exp = (-(x3.powi(2)) * t).exp();
                let term_inner = 1.0 - (1.0 + x2.powi(2)) * term_exp;
                let model_val = (x1 * term_inner).abs();
                y - model_val
            })
            .collect();

        Some(DVector::from_vec(residuals))
    }

    fn jacobian(&self) -> Option<OMatrix<f64, Dyn, U3>> {
        let x1 = self.p[0];
        let x2 = self.p[1];
        let x3 = self.p[2];

        let n = self.t.len();
        // Use generic constructor for Dyn rows and U3 cols
        let mut jacobian = OMatrix::<f64, Dyn, U3>::zeros_generic(Dyn(n), U3);

        for (i, &t) in self.t.iter().enumerate() {
            let term_exp = (-(x3.powi(2)) * t).exp();
            let term_inner = 1.0 - (1.0 + x2.powi(2)) * term_exp;
            let u = x1 * term_inner;
            let sign = u.signum();

            // Partial derivatives of u
            let du_dx1 = term_inner;
            let du_dx2 = x1 * (-2.0 * x2 * term_exp);
            let du_dx3 = x1 * (1.0 + x2.powi(2)) * (2.0 * x3 * t) * term_exp;

            // Jacobian entry J_ij = - sign(u) * du/dp_j
            jacobian[(i, 0)] = -sign * du_dx1;
            jacobian[(i, 1)] = -sign * du_dx2;
            jacobian[(i, 2)] = -sign * du_dx3;
        }

        Some(jacobian)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn make_problem(
        x1: f64,
        x2: f64,
        x3: f64,
        t_vals: &[f64],
        y_vals: &[f64],
    ) -> LookLockerProblem {
        LookLockerProblem {
            t: DVector::from_vec(t_vals.to_vec()),
            y: DVector::from_vec(y_vals.to_vec()),
            p: Vector3::new(x1, x2, x3),
        }
    }

    fn model_val(x1: f64, x2: f64, x3: f64, t: f64) -> f64 {
        (x1 * (1.0 - (1.0 + x2 * x2) * (-(x3 * x3) * t).exp())).abs()
    }

    #[test]
    fn residuals_exact_fit_is_zero() {
        let (x1, x2, x3) = (1.0, 1.118_f64, 0.5); // x2 = sqrt(1.25)
        let t_vals: Vec<f64> = (1..=5).map(|i| i as f64 * 0.1).collect();
        let y_vals: Vec<f64> = t_vals.iter().map(|&t| model_val(x1, x2, x3, t)).collect();
        let prob = make_problem(x1, x2, x3, &t_vals, &y_vals);
        let res = prob.residuals().unwrap();
        for &r in res.iter() {
            assert!(r.abs() < 1e-10, "residual {r} not near zero");
        }
    }

    #[test]
    fn residuals_length_matches_time_points() {
        let prob = make_problem(1.0, 1.0, 1.0, &[0.1, 0.2, 0.3, 0.4], &[0.5, 0.6, 0.7, 0.8]);
        assert_eq!(prob.residuals().unwrap().len(), 4);
    }

    #[test]
    fn jacobian_shape_is_n_by_3() {
        let prob = make_problem(1.0, 1.0, 1.0, &[0.1, 0.2, 0.3], &[0.4, 0.5, 0.6]);
        let jac = prob.jacobian().unwrap();
        assert_eq!(jac.nrows(), 3);
        assert_eq!(jac.ncols(), 3);
    }

    #[test]
    fn set_and_get_params_roundtrip() {
        let p0 = Vector3::new(1.0, 2.0, 3.0);
        let mut prob = make_problem(0.0, 0.0, 0.0, &[0.1], &[0.5]);
        prob.set_params(&p0);
        assert_eq!(prob.params(), p0);
    }

    #[test]
    fn residual_sign_convention() {
        // residual = y - |model|; positive when y > model, negative when y < model
        let t_vals = vec![0.3];
        let mv = model_val(1.0, 1.0, 1.0, 0.3);
        let prob_high = make_problem(1.0, 1.0, 1.0, &t_vals, &[mv + 1.0]);
        let prob_low = make_problem(1.0, 1.0, 1.0, &t_vals, &[mv - 1.0]);
        assert!(prob_high.residuals().unwrap()[0] > 0.0);
        assert!(prob_low.residuals().unwrap()[0] < 0.0);
    }

    #[test]
    fn jacobian_finite_difference_check() {
        let t_vals = vec![0.1, 0.3, 0.5];
        let y_vals = vec![0.2, 0.4, 0.6];
        let (x1, x2, x3) = (1.0, 1.2, 0.8);
        let h = 1e-6;

        let prob = make_problem(x1, x2, x3, &t_vals, &y_vals);
        let jac = prob.jacobian().unwrap();

        for (col, (dx1, dx2, dx3)) in [(h, 0.0, 0.0), (0.0, h, 0.0), (0.0, 0.0, h)]
            .iter()
            .enumerate()
        {
            let r_plus = make_problem(x1 + dx1, x2 + dx2, x3 + dx3, &t_vals, &y_vals)
                .residuals()
                .unwrap();
            let r_minus = make_problem(x1 - dx1, x2 - dx2, x3 - dx3, &t_vals, &y_vals)
                .residuals()
                .unwrap();
            for i in 0..3 {
                let fd = (r_plus[i] - r_minus[i]) / (2.0 * h);
                assert!(
                    (jac[(i, col)] - fd).abs() < 1e-4,
                    "J[{i},{col}] analytic={} fd={fd}",
                    jac[(i, col)]
                );
            }
        }
    }
}
