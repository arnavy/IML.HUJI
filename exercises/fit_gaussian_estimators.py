import pandas

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    X = np.random.normal(10, 1, 1000)

    # Question 1 - Draw samples and print fitted model
    ug = UnivariateGaussian()
    ug.fit(X)
    print("({},{})".format(ug.mu_, ug.var_))

    # Question 2 - Empirically showing sample mean is consistent
    mus = np.zeros(100)
    sample_size = np.arange(10, 1001, 10)
    for i in range(100):
        mus[i] = UnivariateGaussian().fit(X[:sample_size[i]]).mu_

    fig = px.scatter(x=sample_size, y=np.abs(mus - 10), title="Univariate Gaussian - Better With Every Sample",
                     labels={"x": "sample size", "y": "distance of estimated expectation"})
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = ug.pdf(X)
    fig = px.scatter(x=X, y=pdfs, title="Univariate Gaussian - Samples And Their PDF",
                     labels={"x": "samples", "y": "PDFs"})
    fig.show()

def test_multivariate_gaussian():
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(np.array([0, 0, 4, 0]),
                                      cov,
                                      size=1000)
    # Question 4 - Draw samples and print fitted model
    mgu = MultivariateGaussian().fit(X)
    print(mgu.mu_)
    print(mgu.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    ll = np.zeros([200, 200])
    for i in range(f1.shape[0]):
        for j in range(f3.shape[0]):
            ll[i, j] = mgu.log_likelihood(np.array([f1[i], 0, f3[j], 0]).T, cov, X)
    fig = px.imshow(ll, x=f1, y=f3, title="Multivariate Gaussian - Log Likelihood Values",
                    labels={"x": "f1 values", "y": "f3 values"})
    fig.show()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
