// own extension to https://crates.io/crates/average

#[derive(Debug, Clone)]
pub struct CoVariance {
    /// Estimator of X average.
    avg_x: Mean,
    /// Estimator of Y average.
    avg_y: Mean,
    /// Intermediate sum of squares for calculating the covariance.
    sum_2: f64,
    /// Intermediate sum of squares for calculating the X variance.
    sum_2_x: f64,
    /// Intermediate sum of squares for calculating the Y variance.
    sum_2_y: f64,
}

impl CoVariance {
    /// Create a new covariance estimator.
    #[inline]
    pub fn new() -> CoVariance {
        CoVariance { avg_x: Mean::new(), avg_y: Mean::new(), sum_2: 0., sum_2_x: 0., sum_2_y: 0. }
    }

    /// Increment the sample size.
    ///
    /// This does not update anything else.
    #[inline]
    fn increment(&mut self) {
        self.avg_x.increment();
        self.avg_y.increment();
    }

    /// Add an observation given an already calculated difference from the mean
    /// divided by the number of samples, assuming the inner count of the sample
    /// size was already updated.
    ///
    /// This is useful for avoiding unnecessary divisions in the inner loop.
    #[inline]
    fn add_inner(&mut self, delta_x: f64, delta_y: f64) {
        // This algorithm introduced by Welford in 1962 trades numerical
        // stability for a division inside the loop.
        //
        // See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance.
        let n = f64::approx_from(self.avg_x.len()).unwrap();
        self.avg_x.add_inner(delta_x);
        self.avg_y.add_inner(delta_y);

        let n1 = n * (n - 1.);
        self.sum_2 += delta_x * delta_y * n1;
        self.sum_2_x += delta_x * delta_x * n1;
        self.sum_2_y += delta_y * delta_y * n1;
    }

    /// Determine whether the sample is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.avg_x.is_empty() || self.avg_y.is_empty()
    }

    /// Estimate the mean of the X population.
    ///
    /// Returns 0 for an empty sample.
    #[inline]
    pub fn mean_x(&self) -> f64 {
        self.avg_x.mean()
    }

    /// Estimate the mean of the Y population.
    ///
    /// Returns 0 for an empty sample.
    #[inline]
    pub fn mean_y(&self) -> f64 {
        self.avg_y.mean()
    }

    /// Return the sample size.
    #[inline]
    pub fn len(&self) -> u64 {
        self.avg_x.len()
    }

    /// Calculate the sample covariance.
    ///
    /// This is an unbiased estimator of the variance of the population.
    #[inline]
    pub fn sample_covariance(&self) -> f64 {
        let n = self.avg_x.len();
        if n < 2 {
            return 0.;
        }
        self.sum_2 / f64::approx_from(n - 1).unwrap()
    }

    /// Calculate the sample X variance.
    ///
    /// This is an unbiased estimator of the variance of the X population.
    #[inline]
    pub fn sample_variance_x(&self) -> f64 {
        let n = self.avg_x.len();
        if n < 2 {
            return 0.;
        }
        self.sum_2_x / f64::approx_from(n - 1).unwrap()
    }

    /// Calculate the sample Y variance.
    ///
    /// This is an unbiased estimator of the variance of the Y population.
    #[inline]
    pub fn sample_variance_y(&self) -> f64 {
        let n = self.avg_y.len();
        if n < 2 {
            return 0.;
        }
        self.sum_2_y / f64::approx_from(n - 1).unwrap()
    }

    #[inline]
    pub fn add(&mut self, sample_x: f64, sample_y: f64) {
        self.increment();
        let delta_x = (sample_x - self.avg_x.mean())
            / f64::approx_from(self.avg_x.len()).unwrap();
        let delta_y = (sample_y - self.avg_y.mean())
            / f64::approx_from(self.avg_y.len()).unwrap();
        self.add_inner(delta_x, delta_y);
    }
}