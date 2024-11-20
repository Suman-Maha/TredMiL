import numpy as np
from scipy.special import logsumexp
from scipy.stats import truncnorm as t_norm


def em(e, m, X, theta, steps):
    thetas = [theta]
    n = len(X)
    for _ in range(steps):
        Q = e(X, theta)
        theta = m(Q, X)
        thetas.append(theta)
    return thetas


def gmm_joint(X, theta):
    pi, mu, sigma = theta
#    if sigma == 0:
#    	sigma = 1e-10
    diffs = np.subtract.outer(X, mu) / sigma
    quadratic = -0.5 * diffs ** 2
    normalizing = -0.5 * np.log(2 * np.pi) - np.log(sigma)
    logpx_given_z = quadratic + normalizing
    return logpx_given_z  + np.log(pi)


def gmm_e(X, theta):
    joint = gmm_joint(X, theta)
    pz_given_x = np.exp(
        joint - logsumexp(joint, axis=1, keepdims=True))
    return pz_given_x


def gmm_m(Q, X):
    mixture_weight_sums = Q.sum(axis=0)
    pi = mixture_weight_sums / mixture_weight_sums.sum()
    mu = Q.T.dot(X) / mixture_weight_sums
    diffs = np.subtract.outer(X, mu)
    diffs *= Q
    sigma = np.sum(diffs ** 2, axis=0)
    sigma /= mixture_weight_sums
    return pi, mu, np.sqrt(sigma)


def gmm_likelihood(X, theta):
    joint = gmm_joint(X, theta)
    return logsumexp(joint)


def sample_mixture_truncated_gaussian(mean, std, num_samples, left_extreme, right_extreme):
    s_len = len(mean)
    rd_centers = np.random.choice(s_len, num_samples)
    if std == 0:
    	std = 1e-10
    ta = (left_extreme - mean[rd_centers]) / std
    tb = (right_extreme - mean[rd_centers]) / std
    x_samples_tg_mm = t_norm.rvs(ta, tb, loc=mean[rd_centers], scale=std)
    return x_samples_tg_mm


def extract_sample_mixture_truncated_gaussian(mean, std, dim, left_limit, right_limit, batch_size):
    mean = np.array(mean)
    std = np.array(std)
    left_limit = np.array(left_limit)
    right_limit = np.array(right_limit)
    samples = sample_mixture_truncated_gaussian(mean=mean, std=std, num_samples=dim, left_extreme=left_limit,
                                                right_extreme=right_limit)
    sample_extracted = samples.reshape([batch_size, dim, 1, 1])
    #color_sample = torch.tensor(sample_extracted)
    color_sample = sample_extracted
    return color_sample


def extract_mean_variance(color_appearance_code_reshaped):
    #color_appearance_code_reshaped = color_appearance_code.reshape([-1])
    color_appearance_code_reshaped = color_appearance_code_reshaped.cpu().detach().numpy()
    #pi1 = np.random.uniform(0.4, 0.6)
    #theta = (
    #    [pi1, 1 - pi1],
    #    np.percentile(color_appearance_code_reshaped, [25, 75]),
    #    np.random.uniform(1.0, 2, size=2)
    #)
    #pi1 = 0.55
    pi1 = np.random.uniform(0.4, 0.6)
    theta = (
         [pi1, 1 - pi1],
         np.percentile(color_appearance_code_reshaped, [25, 75]),
         [0.5, 0.5]
         )
    thetas = em(gmm_e, gmm_m, color_appearance_code_reshaped, theta, 10)
    mean_stain1 = thetas[-1][1][0]
    mean_stain2 = thetas[-1][1][1]
    var_stain1 = thetas[-1][2][0]
    var_stain2 = thetas[-1][2][1]
    mixing_stain1 = thetas[-1][0][0]
    mixing_stain2 = thetas[-1][0][1]
    return mean_stain1, mean_stain2, var_stain1, var_stain2, mixing_stain1, mixing_stain2







  
