#![cfg_attr(feature = "cargo-clippy", allow(float_cmp, map_clone))]

#[macro_use] extern crate average;

extern crate rand;
extern crate stats;

/// Create a random vector by sampling from a normal distribution.
fn initialize_vec(size: usize) -> Vec<f64> {
    use rand::distributions::{Normal, Distribution};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    let normal = Normal::new(2.0, 3.0);
    let mut values = Vec::with_capacity(size);
    let mut rng = SmallRng::from_seed(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
    for _ in 0..size {
        values.push(normal.sample(&mut rng));
    }
    values
}

#[test]
fn average_vs_streaming_stats_small() {
    let values = initialize_vec(100);
    let a: average::MeanWithError = values.iter().collect();
    let b: stats::OnlineStats = values.iter().map(|x| *x).collect();
    assert_almost_eq!(a.mean(), b.mean(), 1e-16);
    assert_almost_eq!(a.population_variance(), b.variance(), 1e-14);
}

#[test]
fn average_vs_streaming_stats_large() {
    let values = initialize_vec(1_000_000);
    let a: average::MeanWithError = values.iter().collect();
    let b: stats::OnlineStats = values.iter().map(|x| *x).collect();
    assert_almost_eq!(a.mean(), b.mean(), 1e-16);
    assert_almost_eq!(a.population_variance(), b.variance(), 1e-12);
}
