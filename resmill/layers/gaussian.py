import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal

from .base import Layer


class GaussianLayer(Layer):
    """Layer with Sequential Gaussian Simulation (SGS) geology."""

    def create_geology(self, poro_ave, perm_ave, poro_std, perm_std, ntg,
                       facies_filter=(25.0, 50.0, 2.5),
                       sand_filter=(15.0, 25.0, 1.5),
                       nugget=0.05, poro_perm_corr=0.6):
        """Generate geology using Gaussian simulation.

        Parameters
        ----------
        poro_ave, perm_ave : float
            Mean porosity and log10(permeability).
        poro_std, perm_std : float
            Standard deviation of porosity and log10(permeability).
        ntg : float
            Net-to-gross ratio (0 to 1).
        facies_filter : tuple of 3 floats
            Spatial-correlation lengths (x, y, z) for the facies field,
            **in meters**. Converted to cell-sigma internally via
            ``(self.dx, self.dy, self.dz)``.
        sand_filter : tuple of 3 floats
            Spatial-correlation lengths (x, y, z) for the property
            field, **in meters**. Same conversion as ``facies_filter``.
        nugget : float
            Nugget effect for property field.
        poro_perm_corr : float
            Correlation coefficient between porosity and permeability.
        """
        self.poro_ave = poro_ave
        self.perm_ave = perm_ave
        self.poro_std = poro_std
        self.perm_std = perm_std
        self.ntg = ntg
        nx, ny, nz = self.nx, self.ny, self.nz

        # Convert physical (m) correlation lengths to cell-sigma for
        # scipy.ndimage.gaussian_filter (which works in array indices).
        cell_size = (self.dx, self.dy, self.dz)
        facies_sigma = tuple(s / d for s, d in zip(facies_filter, cell_size))
        sand_sigma = tuple(s / d for s, d in zip(sand_filter, cell_size))

        # Facies: filtered Gaussian field thresholded at NTG percentile
        # 3x oversampling to avoid edge effects, crop center
        facies_field = gaussian_filter(
            np.random.normal(0, 1, (3 * nx, 3 * ny, 3 * nz)),
            facies_sigma, mode='wrap'
        )
        facies_field = facies_field[nx:2*nx, ny:2*ny, nz:2*nz]
        self.active = (facies_field < np.percentile(facies_field, ntg * 100)).astype(np.int8)
        # Alluvsim-style facies code: -1 (FF, floodplain / shale) where
        # inactive, 3 (LA = lateral-accretion / bar — closest analogue
        # to a sand-shale heterogeneous body) where active.
        self.facies = np.where(self.active == 1, 3, -1).astype(np.int8)

        # Property field with spatial correlation + nugget
        poro_field = gaussian_filter(
            np.random.normal(0, 1, (3 * nx, 3 * ny, 3 * nz)),
            sand_sigma, mode='wrap'
        )
        poro_nug = np.random.normal(0, 1, (3 * nx, 3 * ny, 3 * nz))
        poro_field = poro_field + nugget * poro_nug
        poro_field = poro_field[nx:2*nx, ny:2*ny, nz:2*nz]

        # Correlated poro-perm samples via bivariate normal
        cov = [[1, poro_perm_corr], [poro_perm_corr, 1]]
        samples = multivariate_normal([0, 0], cov).rvs(int(poro_field.size * 1.2))
        samples[:, 0] = samples[:, 0] * poro_std + poro_ave
        samples[:, 1] = samples[:, 1] * perm_std + perm_ave

        # Filter outliers
        samples = samples[samples[:, 0] < (poro_ave + 5 * poro_std)]
        samples = samples[samples[:, 0] > (poro_ave - 5 * poro_std)]
        samples = samples[samples[:, 1] < (perm_ave + 5 * perm_std)]
        samples = samples[:poro_field.size]

        # Quantile transform: match ranks of spatial field to bivariate samples
        flat = poro_field.flatten()
        order = np.argsort(flat)
        samples = samples[np.argsort(samples[:poro_field.size, 0])]
        flat[order] = samples[:, 0]
        perm_flat = flat.copy()
        perm_flat[order] = samples[:, 1]

        self.poro_mat = flat.reshape(nx, ny, nz) * self.active
        self.perm_mat = (10 ** perm_flat.reshape(nx, ny, nz)) * self.active
