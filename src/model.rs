use levenberg_marquardt::LeastSquaresProblem;
use nalgebra::{DVector, OVector, U3, Dyn, OMatrix};

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

        let residuals: Vec<f64> = self.t.iter().zip(self.y.iter()).map(|(&t, &y)| {
            let term_exp = (-(x3.powi(2)) * t).exp();
            let term_inner = 1.0 - (1.0 + x2.powi(2)) * term_exp;
            let model_val = (x1 * term_inner).abs();
            y - model_val
        }).collect();

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
