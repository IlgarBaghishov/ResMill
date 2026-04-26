import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal

from .base import Layer


class LobeLayer(Layer):
    """Layer with turbidite lobe deposition geology."""

    def create_geology(self, poro_ave, perm_ave, poro_std, perm_std, ntg,
                       dhmin=4, dhmax=4, rmin=42, rmax=44, asp=1.5,
                       theta0=0, m=100, upthinning=True, bouma_factor=0):
        """Generate turbidite lobe geology.

        Parameters
        ----------
        poro_ave, perm_ave : float
            Mean porosity and log10(permeability).
        poro_std, perm_std : float
            Standard deviation of porosity and log10(permeability).
        ntg : float
            Net-to-gross ratio (0 to 1).
        dhmin, dhmax : float
            Lobe height range (grid cells).
        rmin, rmax : float
            Lobe radius range (grid cells).
        asp : float
            Lobe aspect ratio (elongation).
        theta0 : float
            Mean lobe orientation (degrees).
        m : float
            Compensation exponent (higher = more clustered stacking).
        upthinning : bool
            Apply vertical thinning upward.
        bouma_factor : float
            Bouma sequence discretization factor (0 = none).
        """
        self.poro_ave = poro_ave
        self.perm_ave = perm_ave
        self.poro_std = poro_std
        self.perm_std = perm_std
        self.ntg = ntg

        allfacies, allporo, self.allsurface = self._lobemodeling(
            dhmax=dhmax, dhmin=dhmin, rmin=rmin, rmax=rmax, asp=asp,
            theta0=theta0, m=m, upthinning=upthinning, bouma_factor=bouma_factor,
        )
        # Swap axes from (nz, ny, nx) to (nx, ny, nz)
        lobe_poro = np.swapaxes(allporo[-1], 0, -1)

        nx, ny, nz = self.nx, self.ny, self.nz
        sand_filt = [1.5, 2.5, 1.5]
        facies_filt = [2.5, 5, 2.5]
        sand_nug = 0.05
        lambda_perturb = 0.1

        # Facies: lobe structure + Gaussian perturbation, threshold at NTG
        facies_perturb = gaussian_filter(
            np.random.normal(0, 1, (3 * nx, 3 * ny, 3 * nz)),
            facies_filt, mode='wrap',
        )
        facies_perturb = facies_perturb[nx:2*nx, ny:2*ny, nz:2*nz]
        self.active = lobe_poro + lambda_perturb * facies_perturb
        self.active = (self.active > np.percentile(self.active.flatten(), (1 - ntg) * 100))

        # Property field: lobe structure + Gaussian noise
        poro_field = np.random.normal(0, 1, (3 * nx, 3 * ny, 3 * nz))
        poro_nug = np.random.normal(0, 1, (3 * nx, 3 * ny, 3 * nz))
        poro_field = gaussian_filter(poro_field, sand_filt, mode='wrap') + sand_nug * poro_nug
        poro_field = poro_field[nx:2*nx, ny:2*ny, nz:2*nz]
        poro_field = lobe_poro + lambda_perturb * poro_field
        perm_field = poro_field.copy()

        # Quantile transform with correlated bivariate normal
        cov = [[1, 0.6], [0.6, 1]]
        samples = multivariate_normal([0, 0], cov).rvs(int(poro_field.size * 1.2))
        samples[:, 0] = samples[:, 0] * poro_std + poro_ave
        samples[:, 1] = samples[:, 1] * perm_std + perm_ave

        samples = samples[samples[:, 0] < (poro_ave + 5 * poro_std)]
        samples = samples[samples[:, 0] > (poro_ave - 5 * poro_std)]
        samples = samples[samples[:, 1] < (perm_ave + 5 * perm_std)]
        samples = samples[:poro_field.size]

        flat = poro_field.flatten()
        order = np.argsort(flat)
        samples = samples[np.argsort(samples[:poro_field.size, 0])]
        flat[order] = samples[:, 0]
        perm_flat = flat.copy()
        perm_flat[order] = samples[:, 1]

        self.poro_mat = flat.reshape(nx, ny, nz)
        self.perm_mat = perm_flat.reshape(nx, ny, nz)

        self.active = self.active.astype(np.int8)
        self.poro_mat = self.poro_mat * self.active
        self.perm_mat = (10 ** self.perm_mat) * self.active

        # ``lobe_id`` keeps the per-lobe stacking index (1..N) for users
        # who want to colour by lobe generation. ``facies`` is the
        # uniform Alluvsim-style code: -1 (FF, floodplain / shale) where
        # inactive, 3 (LA = lateral-accretion / bar) where the cell
        # belongs to a lobe — closest analogue to Alluvsim's bar
        # facies for a turbidite-lobe deposit.
        if allfacies and len(allfacies) > 0:
            self.lobe_id = np.swapaxes(allfacies[-1].copy(), 0, -1).astype(np.int16)
        else:
            self.lobe_id = np.zeros((nx, ny, nz), dtype=np.int16)
        self.facies = np.where(self.active == 1, 3, -1).astype(np.int8)

    def _lobemodeling(self, dhmax=4, dhmin=4, rmin=42, rmax=44, asp=1.5,
                      theta0=0, m=100, upthinning=True, bouma_factor=0):
        facies = np.zeros((self.nz, self.ny, self.nx))
        poro = facies.copy() - 0.1
        allsurface = []
        surface = 0.000001 * np.ones((self.ny, self.nx))
        surface0 = surface.copy()
        lat_size = self.nx * self.ny
        loc_idx = np.arange(lat_size)
        theta0 = theta0 / 180 * np.pi
        allfacies = []
        allporo = []
        allsurface.append(surface.copy())

        i = 0
        iiii = 10000
        while i < iiii - 1:
            surface0 = surface.copy()
            zz = surface
            prob = (1 / (surface - zz.min() + 0.001)**m) / np.sum(1 / (surface - zz.min() + 0.001))
            prob = prob / np.sum(prob)
            prob_flat = prob.flatten()
            loc = np.random.choice(loc_idx, p=prob_flat)
            y = loc // self.nx
            x = loc - self.nx * y

            theta = theta0 + np.random.normal(0, 20 / 180 * np.pi)
            dh = np.random.uniform(dhmin, dhmax)
            r = np.random.uniform(rmin, rmax)

            self._update_surface(x, y, r, asp, theta, dh, surface)
            if i != 0:
                surface2 = surface.copy()
                surface = surface0 + (surface - surface0) * (1 - (surface0 / surface0.max())**1.2)
                dsurface = surface - surface0
                surface = surface0 + dsurface * (np.sum(surface2 - surface0) / np.sum(dsurface + 0.000000001))

            dz = surface - surface0
            ychange, xchange = np.where(dz > 0)
            allsurface.append(surface.copy())
            self._assign_prop(xchange, ychange, x, y, theta, surface0, surface,
                              facies, r, poro, dz, asp, i + 1, upthinning, bouma_factor)
            i += 1
            if i == iiii or surface.max() >= self.nz + 4:
                allfacies.append(facies.copy())
                allporo.append(poro.copy())
                break
        return allfacies, allporo, allsurface

    def _update_surface(self, x, y, r, asp, theta, dh, surface):
        for ii in np.arange(max(0, int(y - r * asp)), min(int(y + r * asp), self.ny)):
            for jj in np.arange(max(0, int(x - r * asp)), min(int(x + r * asp), self.nx)):
                dx = jj - x
                dy = ii - y
                dx2 = dx * np.cos(theta) - dy * np.sin(theta)
                dy2 = dx * np.sin(theta) + dy * np.cos(theta)
                r1 = np.sqrt((dx2 / asp)**2 + dy2**2)
                if r1**2 <= r**2:
                    dz0 = -dh / (r**2) * (r1**2) + dh
                    surface[ii, jj] = surface[ii, jj] + dz0

    def _assign_prop(self, xchange, ychange, x, y, theta, surface0, surface,
                     facies, r, poro, dz, asp, i, upthinning, bouma_factor):
        for n in range(xchange.size):
            ii = ychange[n]
            jj = xchange[n]
            dx = jj - x
            dy = ii - y
            dx2 = dx * np.cos(theta) - dy * np.sin(theta)
            dy2 = dx * np.sin(theta) + dy * np.cos(theta)
            r1 = np.sqrt((dx2 / asp)**2 + dy2**2)
            bot = int(np.rint(surface0[ii, jj]))
            top = int(min(np.rint(surface[ii, jj]), self.nz))
            # Use continuous surface values for smooth porosity gradients
            surf_bot = surface0[ii, jj]
            surf_top = min(surface[ii, jj], float(self.nz))
            thickness = surf_top - surf_bot
            poromin = 0.05
            poromax = 0.3 * ((1 - surf_bot / self.nz) / 2 + 0.5) + 0.05 if upthinning else 0.35
            if top > bot and thickness > 1e-10:
                facies[bot:top, ii, jj][facies[bot:top, ii, jj] == 0] = i
                for kk in np.arange(bot, top):
                    upthinning_factor = ((1 - kk / self.nz) / 2 + 0.5) if upthinning else 1
                    vert_decay = max((surf_top - kk) / thickness, 0.0)
                    poro[kk, ii, jj] = 0.3 * vert_decay * (1 - (r1 / r)) * upthinning_factor + 0.05
                    poronorm = (poro[kk, ii, jj] - poromin) / (poromax - poromin)
                    bouma_seq_lims = [0, 0.1, 0.2, 0.3, 0.4, 1]
                    for bouma_idx in range(len(bouma_seq_lims) - 1):
                        if bouma_seq_lims[bouma_idx] <= poronorm < bouma_seq_lims[bouma_idx + 1]:
                            bouma_seq_mid = (bouma_seq_lims[bouma_idx] + bouma_seq_lims[bouma_idx + 1]) / 2
                            poronorm = (1 - bouma_factor) * (poronorm - bouma_seq_mid) + bouma_seq_mid
                            break
                    poro[kk, ii, jj] = poromin + poronorm * (poromax - poromin)
