import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


def safe_sqrt(x):
    return x.clamp(min=1e-5).sqrt()


def det2x2(mats):
    dets = mats[...,0,0] * mats[...,1,1] - mats[...,0,1] * mats[...,1,0]
    return dets


def trace2x2(mats):
    return mats[...,0,0] + mats[...,1,1]


def inv2x2(mats):
    invs = torch.empty_like(mats)
    invs[...,0,0] = mats[...,1,1]
    invs[...,1,1] = mats[...,0,0]
    invs[...,0,1] = -mats[...,1,0]
    invs[...,1,0] = -mats[...,0,1]

    determinants = det2x2(mats)

    invs = invs * (1.0 / determinants[..., None, None])
    
    return invs, determinants


def cholesky2x2(mats, upper=True):
    chol = torch.empty_like(mats)
    if upper:
        chol[...,1,0] = 0
        chol[...,0,0] = torch.sqrt(mats[...,0,0])
        chol[...,0,1] = torch.div(mats[...,1,0], chol[...,0,0])
        chol[...,1,1] = torch.sqrt(mats[...,1,1] - torch.square(chol[...,0,1]))
    else:
        chol[...,0,1] = 0
        chol[...,0,0] = torch.sqrt(mats[...,0,0])
        chol[...,1,0] = torch.div(mats[...,1,0], chol[...,0,0])
        chol[...,1,1] = torch.sqrt(mats[...,1,1] - torch.square(chol[...,1,0]))
    
    return chol


def chol_log_det(L):
    return 2*torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=1)


def trace(A):
    return torch.sum(torch.diagonal(A, dim1=-2, dim2=-1), dim=1)


def quadratic(x, A):
    x_sq = torch.square(x)
    x_corr = x[...,0]*x[...,1]
    xtAx = A[...,0,0]*x_sq[...,0] + 2*A[...,0,1]*x_corr + A[...,1,1]*x_sq[...,1]
    return xtAx


# https://etd.ohiolink.edu/apexprod/rws_etd/send_file/send?accession=osu1437409380&disposition=attachment
def nonstationary(x1, E1, x2, E2):
    diff = (x1.unsqueeze(2) - x2.unsqueeze(1)).float()
    E_sum = E1.unsqueeze(2) + E2.unsqueeze(1)
    E_sum_inv, E_sum_det = inv2x2(E_sum)
    Q = 0.5 * quadratic(diff, E_sum_inv)

    E1_det_quarter_root = torch.sqrt(torch.sqrt(torch.linalg.det(E1)))
    E2_det_quarter_root = torch.sqrt(torch.sqrt(torch.linalg.det(E2)))
    C = 2.0 * E1_det_quarter_root.unsqueeze(2) * E2_det_quarter_root.unsqueeze(1) / torch.sqrt(E_sum_det)

    return Q, C


# https://www.jmlr.org/papers/volume5/jebara04a/jebara04a.pdf
# bhattacharyya kernel: p=0.5 gives K(x,x) = 1
# expected likelihood kernel: p=1.0
# Assumess D=2, p=0.5
def prob_product_quad(x1, E1, x2, E2):
    dim1 = len(x1.shape)-1
    dim2 = len(x2.shape)-2
    diff = (x1.unsqueeze(dim1) - x2.unsqueeze(dim2)).float()

    Q = (E1[...,1,1].unsqueeze(dim1) + E2[...,1,1].unsqueeze(dim2)) * torch.square(diff[...,0])
    Q = Q - 2*(E1[...,0,1].unsqueeze(dim1) + E2[...,0,1].unsqueeze(dim2)) * diff[...,0] * diff[...,1]
    Q = Q + (E1[...,0,0].unsqueeze(dim1) + E2[...,0,0].unsqueeze(dim2)) * torch.square(diff[...,1])
    E_sum_det = (E1[...,0,0].unsqueeze(dim1) + E2[...,0,0].unsqueeze(dim2)) * (E1[...,1,1].unsqueeze(dim1) + E2[...,1,1].unsqueeze(dim2)) - torch.square(E1[...,0,1].unsqueeze(dim1) + E2[...,0,1].unsqueeze(dim2))
    Q = Q / E_sum_det 

    return Q / 2.0
  

# Assumes D=2, p=0.5
def prob_product_constant(E1, E2):
    dim1 = len(E1.shape)-2
    dim2 = len(E2.shape)-3

    E1_det_root = det2x2(E1) ** (0.25)
    E2_det_root = det2x2(E2) ** (0.25)
    C = 2.0 * E1_det_root.unsqueeze(dim1) * E2_det_root.unsqueeze(dim2) / safe_sqrt((E1[...,0,0].unsqueeze(dim1) + E2[...,0,0].unsqueeze(dim2)) * (E1[...,1,1].unsqueeze(dim1) + E2[...,1,1].unsqueeze(dim2)) - torch.square(E1[...,0,1].unsqueeze(dim1) + E2[...,0,1].unsqueeze(dim2)))  

    return C


## Diagonal covariance functions
def diagonal_nonstationary(coords, E):
    K_diag = torch.ones(coords.shape[0], coords.shape[1], device=coords.device)
    return K_diag


def diagonal_prob_product(coords, E):
    E_det_root = torch.sqrt(det2x2(E))
    E_sum_det = det2x2(2*E)
    C = 2.0 * E_det_root / safe_sqrt(E_sum_det)
    Q = torch.zeros_like(C)
    return Q, C


## Isotropic covariance functions
def squared_exponential(Q):
    K = torch.exp(-0.5*Q)
    return K


def matern(Q):
    Q_sqrt = safe_sqrt(Q) # Constant term for stability, otherwise nan on backward
    # v=3/2
    tmp = (np.sqrt(3))*Q_sqrt
    k_v_3_2 = (1 + tmp) * torch.exp(-tmp)

    K = k_v_3_2
    return K


# Construct convolution of heteregenous Gaussian kernels on R2 by Chris Paciorek thesis
def convolutionOfGaussianCovariance(x1, E1, x2, E2):

    diff = (x1.unsqueeze(2) - x2.unsqueeze(1)).float()

    E_sum = E1.unsqueeze(2) + E2.unsqueeze(1)
    E_sum_inv, E_sum_det = inv2x2(E_sum)

    diff_sq = torch.square(diff)
    diff_corr = diff[...,0]*diff[...,1]
    Q = 0.5 * (E_sum_inv[...,0,0]*diff_sq[...,0] + 2*E_sum_inv[...,0,1]*diff_corr + E_sum_inv[...,1,1]*diff_sq[...,1])

    k = x1.shape[-1]
    C = 1.0/torch.sqrt( ((2*np.pi)**k) * E_sum_det)
    K = C * torch.exp(-Q)

    return K


def gaussianKernel(x1, E1, x2):

    diff = (x1.unsqueeze(2) - x2.unsqueeze(1)).float()
    E1_inv, E1_det = inv2x2(E1)

    diff_sq = torch.square(diff)
    diff_corr = diff[...,0]*diff[...,1]
    Q = 0.5*(E1_inv[...,0,0].unsqueeze(-1)*diff_sq[...,0] + 2*E1_inv[...,0,1].unsqueeze(-1)*diff_corr + E1_inv[...,1,1].unsqueeze(-1)*diff_sq[...,1])
        
    k = x1.shape[-1]
    C = 1.0/safe_sqrt( ((2*np.pi)**k) * E1_det)
    K = C.unsqueeze(2) * torch.exp(-Q)

    return K


def diagonalCorrelation(E):
    C = torch.ones(E.shape[0], E.shape[1], device=E.device)
    return C


def diagonalConvolutionOfGaussianCovariance(E):
    E_det = det2x2(E)
    twopi = (2 * np.pi) ** E.shape[-1]
    C = 0.5 / safe_sqrt(twopi * E_det)
    return C


def diagonalGaussianKernel(E):
    E_det = det2x2(E)
    twopi = (2 * np.pi) ** E.shape[-1]
    C = 1.0 / safe_sqrt(twopi * E_det)
    return C


def get_kernel_mats_cov(kernel_params):
    device = kernel_params.device
    b, n, _ = kernel_params.shape

    E = torch.empty((b, n, 2, 2), device=device, dtype=kernel_params.dtype)
    E[:,:,0,0] = kernel_params[:,:,0]
    E[:,:,1,1] = kernel_params[:,:,1]
    E[:,:,0,1] = kernel_params[:,:,2].clone()
    E[:,:,1,0] = kernel_params[:,:,2].clone()
    return E


def kernel_params_to_covariance(kernel_img_norm: torch.Tensor):
    B = kernel_img_norm.shape[0]
    C = kernel_img_norm.shape[1]
    H = kernel_img_norm.shape[2]
    W = kernel_img_norm.shape[3]

    kernel_img_tmp = kernel_img_norm.reshape(B, 3, -1).permute(0, 2, 1)
    E = get_kernel_mats_cov(kernel_img_tmp)
    E = E.reshape(B, H, W, 4).permute(0, 3, 1, 2)
    return E


def interpolate_kernel_params(kernel_img, x):
    # x coordinates are normalized [-1,1]
    # NOTE: grid_sample expects (x,y) as in image coordinates (so column then row)
    x_samples = torch.unsqueeze(x, dim=1)
    ind_swap = torch.tensor([1, 0], device=kernel_img.device)
    x_samples = torch.index_select(x_samples, 3, ind_swap)

    assert(kernel_img.shape[1] == 4)

    # kernel_image shape: B x 3 x H x W
    # x shape: B x N x 2
    # output shape: B x 3 x N
    B = kernel_img.shape[0]
    N = x.shape[1]

    # Get sampled features
    sampled_params = F.grid_sample(kernel_img, x_samples, mode='bilinear', padding_mode='reflection', align_corners=False)
    sampled_params = sampled_params.squeeze(dim=2).permute(0, 2, 1)

    kernel_mats = sampled_params.reshape(B, N, 2, 2)
    return kernel_mats


def normalize_coordinates(x_pixel, dims):
  A = 1.0/torch.as_tensor(dims, device=x_pixel.device, dtype=x_pixel.dtype)
  x_norm = 2*A*x_pixel + A - 1
  return x_norm


def sample_coords(depth, num_samples = 2 ** 14, is_training = True):
    B, C, H, W = depth.shape
    device = depth.device
    depth = depth.reshape(B, -1)
    coords_image = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    coords_image_flat = torch.stack(coords_image, dim=-1).unsqueeze(0).repeat(B, 1, 1, 1).reshape(B, -1, 2)
    depth_flat = depth.reshape(B, -1)
    depth_mask = depth_flat > 0

    # Take minimum so that batch sizes are equal
    num_valid = torch.min(torch.count_nonzero(depth_mask, dim=1))
    if is_training:
        num_valid = min(num_valid, num_samples)

    indices = torch.multinomial(depth_mask.float(), num_valid, replacement=False)

    coord_samples = coords_image_flat[:, indices]
    depth_samples = depth_flat[:, indices].unsqueeze(-1)

    return coord_samples, depth_samples


class CovarianceModule(nn.Module):
    def __init__(self, iso_cov_fn, scale_param, scale_prior):
        super().__init__()
        self.iso_cov_fn = iso_cov_fn
        self.scale_param = scale_param
        self.scale_prior = scale_prior

    def get_scale(self):
        return self.scale_prior * torch.exp(self.scale_param)

    def forward(self, coords, E):    
        K = self.iso_cov_fn(prob_product_quad(coords, E, coords, E))
        K_scaled = K * prob_product_constant(E, E)
        K_scaled = K_scaled * self.get_scale()
        return K_scaled

    
class CrossCovarianceModule(CovarianceModule):
    def __init__(self, iso_cov_fn, scale_param, scale_prior):
        super().__init__(iso_cov_fn, scale_param, scale_prior)

    def forward(self, coords_train, E_train, coords_test, E_test):
        K_train_test = self.iso_cov_fn(prob_product_quad(coords_train, E_train, coords_test, E_test))
        K_train_test_scaled = K_train_test * prob_product_constant(E_train, E_test)
        K_train_test_scaled = K_train_test_scaled * self.get_scale()
        return K_train_test_scaled


class DiagonalCovarianceModule(CovarianceModule):
    def __init__(self, iso_cov_fn, scale_param, scale_prior):
        super().__init__(iso_cov_fn, scale_param, scale_prior)

    def forward(self, coords, E):
        Q, C = diagonal_prob_product(coords, E)
        K_diag = C * self.iso_cov_fn(Q)
        K_diag_scaled = K_diag * self.get_scale()
        return K_diag_scaled

 
class GpVfeModuleConstantNoise(nn.Module):
    def __init__(self):
        super().__init__()

    def solve_optimal_mean(self, y, L_A, K_mn, K_mn_y):
        n = y.shape[1]
        
        L_inv_Kmn_y = torch.linalg.solve_triangular(L_A, K_mn_y, upper=False)
        Kmn_sum = torch.sum(K_mn, dim=(2), keepdim=True)
        L_inv_Kmn_sum = torch.linalg.solve_triangular(L_A, Kmn_sum, upper=False)

        A1T_A1 = n
        A1T_b1 = torch.sum(y, dim=(1), keepdim=True)
        tmp = torch.transpose(L_inv_Kmn_sum, dim0=1, dim1=2)
        A2T_A2 = torch.matmul(tmp, L_inv_Kmn_sum)
        A2T_b2 = torch.matmul(tmp, L_inv_Kmn_y)

        mean = (A1T_b1 - A2T_b2) / (A1T_A1 - A2T_A2)
        return mean

    # Assumes variance is a scalar
    def forward(self, K_nn, K_nm, K_mm_diag, depth_gt, mean, var):
        B = K_nn.shape[0]
        num_induced = K_nm.shape[-2]
        # resample K_nm, K_nm, K_mm_diag based on where depth_gt is >0, take min over samples and mask
        mask = (depth_gt > 0).flatten(1)
        num_points = torch.sum(mask, dim=1).min()
        # select num_pts among depth_gt > 0
        depth_gt = depth_gt[:, mask]

        K_nm = K_nm.permute(0,2,1)[mask].permute(0,2,1)
        K_mm_diag = K_mm_diag[mask]
        depth_gt = depth_gt[mask]

        var_inv = 1.0 / var
        def select_random_locations(original_tensor, mask_tensor, N):
            batch_size = original_tensor.shape[0]
            num_points = original_tensor.shape[1]

            selected_locations = []
            for i in range(batch_size):
                mask = mask_tensor[i]  # Get the mask for the current batch
                indices = torch.nonzero(mask.flatten(), as_tuple=False)  # Find the indices where the mask is True
                selected_indices = torch.randperm(indices.shape[0])[:N]  # Randomly select N indices
                selected_locations.append(indices[selected_indices])  # Append the selected indices to the list

            selected_locations = torch.stack(selected_locations)  # Stack the selected indices across batches

            selected_tensor = original_tensor.gather(1, selected_locations.unsqueeze(-1).expand(-1, -1, original_tensor.shape[-1]))

            return selected_tensor

        jitter = 1e-4 * torch.ones(B, num_induced, device=K_nn.device)
        K_nn = K_nn + torch.diag_embed(jitter)
        # Cholesky for inverses
        L_mm, info_mm = torch.linalg.cholesky_ex(K_nn, upper=False)
        A = var * K_nn + torch.matmul(K_nm, torch.transpose(K_nm, dim0=-2, dim1=-1))
        L_A, info_A = torch.linalg.cholesky_ex(A, upper=False)

        info = info_mm + info_A

        with torch.no_grad():
            K_mn_y = torch.matmul(K_nm, depth_gt)
            mean = self.solve_optimal_mean(depth_gt, L_A, K_nm, K_mn_y)

        y_centered = depth_gt - mean
        K_mn_y_centered = torch.matmul(K_nm, y_centered)

        data_term = var_inv * (torch.mean(y_centered * y_centered, dim=1).squeeze() - torch.matmul(torch.transpose(K_mn_y_centered, dim0=-2, dim1=-1), torch.cholesky_solve(K_mn_y_centered, L_A)).squeeze() / num_points)
        complexity_term = ((num_points - num_induced) * torch.log(var) + chol_log_det(L_A) - chol_log_det(L_mm)) / num_points
        trace_term = var_inv * (torch.mean(K_mm_diag, dim=1) - torch.sum(K_nm / num_points * torch.cholesky_solve(K_nm, L_mm), dim=(1,2)))

        neg_log_marg_likelihood_mean = data_term + complexity_term + trace_term

        return neg_log_marg_likelihood_mean, info


class GpSparseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, K_nn, y_train, mean, var):
        
        # Center data using mean
        y_centered = y_train - mean

        A = K_nn + torch.diag_embed(var) # (B, train_points, train_points)
        L, info = torch.linalg.cholesky_ex(A, upper=False, check_errors=False) # (B, train_points, train_points)
        alpha = torch.cholesky_solve(y_centered, L, upper=False)

        # Marginal likelihood
        n = L.shape[-1]
        data_fit = torch.sum(y_centered * alpha, dim=1) # Shape of (batch_size, output_dim)
        data_term = 0.5 * torch.sum(data_fit, dim=1) # Shape of (batch size)
        complexity_term = torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=1)
        constant_term = 0.5 * n * np.log(2*np.pi)
        neg_log_marg_likelihood = data_term + complexity_term + constant_term

        return L, alpha, neg_log_marg_likelihood, info


class GpDenseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, L, alpha, K_nm, K_mm_diag, mean, var=None):

        pred_mean = mean + torch.sum(alpha.unsqueeze(-2) * K_nm.unsqueeze(-1), dim=1)

        v = torch.cholesky_solve(K_nm, L, upper=False)
        pred_var = K_mm_diag - torch.sum(K_nm * v, dim=1)
        pred_var = pred_var.unsqueeze(-1)

        if var is not None:
            pred_var += var

        return pred_mean, pred_var



class NonstationaryGpModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_depth_var_prior = 1e-2
        kernel_scale_prior = 1e0

        # Covariance modules and parameters
        iso_cov_fn = matern
        self.log_depth_var_scale = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        kernel_scale = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.cov_module = CovarianceModule(iso_cov_fn=iso_cov_fn, scale_param=kernel_scale, scale_prior=kernel_scale_prior)
        self.cross_cov_module = CrossCovarianceModule(iso_cov_fn=iso_cov_fn, scale_param=kernel_scale, scale_prior=kernel_scale_prior)
        self.diagonal_cov_module = DiagonalCovarianceModule(iso_cov_fn=iso_cov_fn, scale_param=kernel_scale, scale_prior=kernel_scale_prior)

        # GP Modules
        self.gp_sparse_module = GpSparseModule()
        self.gp_dense_module = GpDenseModule()
        self.gp_vfe_module = GpVfeModuleConstantNoise()

    def get_var(self,):
        return self.log_depth_var_prior * torch.exp(self.log_depth_var_scale)

    def get_scale(self,):
        return self.cov_module.get_scale()

    def condition_level(self, gaussian_covs, sparse_coords_norm, sparse_depth, mean_depth, test_size):
        device = gaussian_covs.device
        b = gaussian_covs.shape[0]
        
        # Unnormalized coords must be in train image reference frame!
        dense_coords = get_test_coords(test_size, device, batch_size=b)
        dense_coords_norm = normalize_coordinates(dense_coords, test_size)

        sparse_vars = self.get_var() * torch.ones_like(sparse_depth)
        sparse_vars = sparse_vars.squeeze(-1)

        E_n = interpolate_kernel_params(gaussian_covs, sparse_coords_norm)
        E_m = interpolate_kernel_params(gaussian_covs, dense_coords_norm)

        K_nn = self.cov_module(sparse_coords_norm, E_n)
        K_nm = self.cross_cov_module(sparse_coords_norm, E_n, dense_coords_norm, E_m)
        K_mm_diag = self.diagonal_cov_module(dense_coords_norm, E_m)

        L, alpha, _, info = self.gp_sparse_module(K_nn, sparse_depth, mean_depth, sparse_vars)
        if info.any():
            print("Cholesky failed")
        pred_depth, pred_var = self.gp_dense_module(L, alpha, K_nm, K_mm_diag, mean_depth)
        pred_depth = torch.permute(pred_depth, (0,2,1))
        pred_depth = torch.reshape(pred_depth, (b,1,test_size[0],test_size[1]))
        pred_var = torch.permute(pred_var, (0,2,1))
        pred_var = torch.reshape(pred_var, (b,1,test_size[0],test_size[1]))

        return pred_depth, pred_var, L, E_n

    def get_covariance(self, gaussian_covs, coords_norm, E=None):
        if E is None:
            E = interpolate_kernel_params(gaussian_covs, coords_norm)
        K = self.cov_module(coords_norm, E)
        return K, E

    def get_covariance_with_noise(self, gaussian_covs, coords_norm, E=None):
        K, E = self.get_covariance(gaussian_covs, coords_norm, E=E)
        b, m, _ = K.shape
        noise = self.get_var() * torch.ones(b, m, device=K.device)
        K = K + torch.diag_embed(noise)
        return K, E

    def get_covariance_chol(self, gaussian_covs, coords_norm):
        K, E = self.get_covariance_with_noise(gaussian_covs, coords_norm)
        L, info = torch.linalg.cholesky_ex(K, upper=False)
        return L, E

    def get_cross_covariance(self, gaussian_covs, coords1_norm, coords2_norm, E1=None, E2=None):
        if E1 is None:
            E1 = interpolate_kernel_params(gaussian_covs, coords1_norm)
        if E2 is None:
            E2 = interpolate_kernel_params(gaussian_covs, coords2_norm)

        K_mn = self.cross_cov_module(coords1_norm, E1, coords2_norm, E2)
        return K_mn

    def get_diagonal_covariance(self, gaussian_covs, coords_norm, E=None):
        if E is None:
            E = interpolate_kernel_params(gaussian_covs, coords_norm)
        K_diag = self.diagonal_cov_module(coords_norm, E)
        return K_diag
  
    def get_correlation_map(self, gaussian_covs, coords_m_norm):
        b, _, h, w = gaussian_covs[-1].shape
        device = gaussian_covs[-1].device 
        coords_n = get_test_coords(test_size, device, batch_size=b)
        coords_n_norm = normalize_coordinates(coords_n, test_size)

        E_m = interpolate_kernel_params(gaussian_covs, coords_m_norm)
        E_n = interpolate_kernel_params(gaussian_covs, coords_n_norm)

        K_mn = self.cross_cov_module(coords_m_norm, E_m, coords_n_norm, E_n)
        K_m_map = torch.reshape(K_mn, (b, h, w))
        return K_m_map

    def get_linear_predictor(self, gaussian_covs, coords_m_norm, coords_n_norm):
        L_mm, E_m = self.get_covariance_chol(gaussian_covs, coords_m_norm)
        E_n = interpolate_kernel_params(gaussian_covs, coords_n_norm)

        K_mn = self.cross_cov_module(coords_m_norm, E_m, coords_n_norm, E_n)
        Kmminv_Kmn = torch.cholesky_solve(K_mn, L_mm, upper=False)
        Knm_Kmminv = torch.transpose(Kmminv_Kmn, dim0=-2, dim1=-1)
        
        return Knm_Kmminv, L_mm, E_m

    def get_kernels(self, gaussian_covs, coord_sparse, coord_dense):
        E_n = gaussian_covs[:, :, coord_sparse[..., 0], coord_sparse[..., 1]]
        E_n = E_n.reshape(E_n.shape[0], E_n.shape[1], 2, 2)
        E_m = gaussian_covs[:, :, coord_dense[..., 0], coord_dense[..., 1]]
        E_m = E_m.reshape(E_m.shape[0], E_m.shape[1], 2, 2)

        with torch.no_grad():
            coord_sparse_norm = normalize_coordinates(coord_sparse, gaussian_covs.shape[-2:])
            coord_dense_norm = normalize_coordinates(coord_dense, gaussian_covs.shape[-2:])

        K_nn = self.cov_module(coord_sparse_norm, E_n)
        K_nm = self.cross_cov_module(coord_sparse_norm, E_n, coord_dense_norm, E_m)
        K_mm_diag = self.diagonal_cov_module(coord_dense_norm, E_m)
        return K_nn, K_nm, K_mm_diag

    @torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
    def forward(self, gaussian_cov_params, sparse_depth, sparse_locations, depth_gt = None):
        B, _, H, W = gaussian_cov_params.shape
        
        # Network
        gaussian_covs = kernel_params_to_covariance(gaussian_cov_params)
        var = self.get_var()
        if self.training:
            dense_locations, dense_depth = sample_coords(depth_gt, is_training=True)
        else:
            dense_locations = torch.meshgrid(torch.arange(H, device=gaussian_cov_params.device), torch.arange(W, device=gaussian_cov_params.device))
            dense_locations = torch.stack(dense_locations, dim=-1).unsqueeze(0).repeat(B, 1, 1, 1).reshape(B, -1, 2)
            dense_depth = depth_gt
        
        K_nn, K_nm, K_mm_diag = self.get_kernels(gaussian_covs, sparse_locations, dense_locations)
        sparse_vars = var * torch.ones_like(sparse_depth)
        sparse_vars = sparse_vars.squeeze(-1)
        
        # do induced ops
        L, info = torch.linalg.cholesky_ex(K_nn + torch.diag_embed(var), upper=False, check_errors=False) # (B, train_points, train_points)
        alpha = torch.cholesky_solve(sparse_depth, L, upper=False)
        if info.any(): print("Cholesky failed for pred_depth!")
        
        # get project ops
        pred_depth = torch.sum(alpha.unsqueeze(-2) * K_nm.unsqueeze(-1), dim=1)
        v = torch.cholesky_solve(K_nm, L, upper=False)
        pred_var = (K_mm_diag - torch.sum(K_nm * v, dim=1)).unsqueeze(-1)

        pred_depth = pred_depth.permute(0, 2, 1).reshape(B, 1, H, W)
        pred_var = pred_var.permute(0, 2, 1).reshape(B, 1, H, W)

        # resample test and Knm, Kmm based on where obs are present in depth_test
        dense_depth_log = torch.log(dense_depth.clamp(min=1e-5))
        dense_depth_log = (dense_depth_log - torch.mean(dense_depth_log, dim=1, keepdim=True) ) / torch.std(dense_depth_log, dim=1, keepdim=True)
        marginal_nll, info = self.gp_vfe_module(K_nn, K_nm, K_mm_diag, dense_depth_log, 0.0, var)

        with torch.no_grad():
            params_extrema = torch.tensor([
                torch.min(gaussian_cov_params[:,0,:,:]).item(), torch.max(gaussian_cov_params[:,0,:,:]).item(), torch.mean(gaussian_cov_params[:,0,:,:]).item(),\
                torch.min(gaussian_cov_params[:,1,:,:]).item(), torch.max(gaussian_cov_params[:,1,:,:]).item(), torch.mean(gaussian_cov_params[:,1,:,:]).item(),\
                torch.min(gaussian_cov_params[:,2,:,:]).item(), torch.max(gaussian_cov_params[:,2,:,:]).item(), torch.mean(gaussian_cov_params[:,2,:,:]).item() \
            ])

        # LOGGING
        if info.any():
            print("Cholesky failed")
            print(params_extrema)
        elif marginal_nll.isnan().any():
            print("NaN NLML found")
            print(params_extrema)
        if marginal_nll.isnan().any():
            print("NLL found nan")
            print(params_extrema)
            return None

        return pred_depth, pred_var, marginal_nll




#   @staticmethod
#   def get_chol_features(K_mn, L):
#     nystrom = torch.linalg.solve_triangular(L, K_mn, upper=False)
#     return nystrom
  
#   @staticmethod
#   def get_nystrom_features(K_mn, K_mm):
#     s, Q = torch.linalg.eigh(K_mm)
#     D_inv_sqrt = torch.diag_embed(1.0/torch.sqrt(s))
#     nystrom = torch.matmul(torch.transpose(Q, dim0=-2, dim1=-1), K_mn)
#     nystrom = torch.matmul(D_inv_sqrt, nystrom)    
#     return nystrom

#   @staticmethod
#   def solve_compact_depth_hierarchical(L_mm, Knm_Kmminv, mean_log_depth, log_depth_obs, log_depth_stdev_inv):
#     batch_size, n, m = Knm_Kmminv.shape
#     device = Knm_Kmminv.device

#     A = torch.empty((batch_size, m+n, m), device=device)
#     identity = torch.eye(m, device=device).reshape((1,m,m)).repeat(batch_size,1,1)
#     L_inv = torch.linalg.solve_triangular(L_mm, identity, upper=False)
#     A[:,:m,:] = L_inv
#     A[:,m:,:] = log_depth_stdev_inv*Knm_Kmminv

#     b = torch.empty((batch_size, m+n, 1), device=device)
#     b[:,:m,:] = torch.linalg.solve_triangular(L_mm, mean_log_depth*torch.ones((batch_size, m, 1), device=device), upper=False)
#     b[:,m:,:] = log_depth_stdev_inv*(log_depth_obs + torch.sum(Knm_Kmminv * mean_log_depth, dim=(2), keepdim=True) - mean_log_depth)

#     compact_log_depth, _, _, _ = torch.linalg.lstsq(A, b)

#     residuals = torch.matmul(A, compact_log_depth) - b
#     total_err = torch.sum(torch.square(residuals), dim=(1,2))

#     return compact_log_depth, total_err

#   @staticmethod
#   def solve_compact_depth(Knm_Kmminv, log_depth_obs, mean_log_depth):
#     A = Knm_Kmminv
#     b = log_depth_obs + torch.sum(Knm_Kmminv * mean_log_depth, dim=(2), keepdim=True) - mean_log_depth
#     compact_log_depth, _, _, _ = torch.linalg.lstsq(A, b)

#     residuals = torch.matmul(A, compact_log_depth) - b
#     total_err = torch.sum(torch.square(residuals), dim=(1,2))
    
#     return compact_log_depth, total_err

#   @staticmethod
#   def solve_mean_depth(Knm_Kmminv, log_depth_obs, sparse_log_depth):
#     A = 1.0 - torch.sum(Knm_Kmminv, dim=(2), keepdim=True)
#     b = log_depth_obs - torch.matmul(Knm_Kmminv, sparse_log_depth)

#     mean_log_depth, _, _, _ = torch.linalg.lstsq(A, b)

#     residuals = torch.matmul(A, mean_log_depth) - b
#     total_err = torch.sum(torch.square(residuals), dim=(1,2))

#     return mean_log_depth, total_err

#   @staticmethod
#   def solve_compact_and_mean_depth(Knm_Kmminv, log_depth_obs):
#     b, n, m = Knm_Kmminv.shape
#     device = Knm_Kmminv.device

#     A = torch.empty((b,n,m+1), device=device)
#     A[:,:,:m] = Knm_Kmminv
#     A[:,:,m:] = 1.0 - torch.sum(Knm_Kmminv, dim=(2), keepdim=True)
#     x, _, _, _ = torch.linalg.lstsq(A, log_depth_obs)
#     compact_log_depth = x[:,:m,:]
#     mean_log_depth = x[:,m:,:]

#     residuals = torch.matmul(A, x) - log_depth_obs
#     total_err = torch.sum(torch.square(residuals), dim=(1,2))

#     return compact_log_depth, mean_log_depth, total_err

#   @staticmethod
#   def woodbury(K_mm, K_mn, var):
#     b, m, n = K_mn.shape
#     var_inv = 1.0/var

#     A = var * K_mm + torch.matmul(K_mn, torch.transpose(K_mn, dim0=-2, dim1=-1))
#     L_A, info_A = torch.linalg.cholesky_ex(A, upper=False)
#     K_nn_approx = torch.matmul(
#         torch.transpose(K_mn, dim0=-2, dim1=-1), 
#         torch.cholesky_solve(K_mn, L_A))
#     K_nn_approx += torch.diag_embed(torch.ones(b,n,device=K_mm.device))
#     K_nn_approx *= var_inv
#     return 