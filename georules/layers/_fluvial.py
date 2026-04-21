"""Fluvial channel simulation engine (private module).

Ported from legacy channel_model/fluvial_3d.py with import fixes.
All physics preserved unchanged.
"""
import numpy as np
from scipy.interpolate import UnivariateSpline
import scipy.signal
from scipy.ndimage import gaussian_filter

from ._genchannel import genchannel
from ._make_cutoff import make_cutoff


class fluvial:
    # facies: 0 shale; 1 pointbar/channel fill; 2 abandon channel; 3 levee; 4 splay

    def __init__(self, station=1/32, complex_wid=500, b=80, Cf=0.0009, A=10.0,
                 I=0.008, Q=0.9, meander_scale=0.8, dwratio=0.4, nlevel=6, pavul=0,
                 nx=256, ny=128, nz=64, xmn=8, ymn=8, xsiz=16, ysiz=16, zsiz=3,
                 rs=69069, ntg=1000, erode=0.1, lsplay=0, msplay=0, hsplay=0,
                 ntime=10, azi0=0, bankratio=2, myidx=0,
                 migration_distance_ratio=1.0, boundary_reflect=True,
                 aggradation_mode='discrete', level_reseed_prob=0.6,
                 level_jump_ratio=1.0):
        self.myidx = myidx
        self.b = b
        self.dwratio = dwratio
        self.LV_asym = 0.9
        self.lV_height = 5 * zsiz
        dz = int((self.b * self.dwratio) / zsiz)

        ddd = 7
        self.aggrad = [
            dz / ddd / np.random.uniform(3, 4),
            dz / ddd / np.random.uniform(3, 4),
            dz / ddd / np.random.uniform(4, 5),
            dz / ddd / np.random.uniform(5, 6),
            dz / ddd / np.random.uniform(5, 6),
            dz / ddd / np.random.uniform(5, 6),
            dz / ddd / np.random.uniform(7, 8),
            dz / ddd / np.random.uniform(7, 8),
            dz / ddd / np.random.uniform(7, 8),
            dz / ddd / np.random.uniform(8, 9),
            dz / ddd / np.random.uniform(9, 10),
            dz / ddd / 10, dz / ddd / 10, dz / ddd / 10,
        ]

        self.bankratio = bankratio
        self.poro0 = 0.3
        self.poro = 0.05 * np.ones((nx, ny, nz))
        self.splay_dist = b * 4
        self.trunc_len = 2 * b
        self.A = A
        self.Cf = Cf
        self.I = I
        self.Q = Q
        self.azi0 = 0
        self.meander_scale = meander_scale
        self.nlevel = nlevel
        self.pavul = pavul
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.xmn = xmn
        self.ymn = ymn
        self.xsiz = xsiz
        self.ysiz = ysiz
        self.zsiz = zsiz
        self.rs = rs
        self.ntg = ntg
        self.erode = erode
        self.lsplay = lsplay
        self.msplay = msplay
        self.hsplay = hsplay
        self.ntime = ntime
        # Per-step lateral migration scale, expressed as a multiple of
        # (xsiz+ysiz). Alluvsim draws distMigrate ~ Gaussian(25-50 m) per
        # step and rescales the bank-velocity field to match. The original
        # hard-coded value here was 2.5 which, combined with the continuous
        # aggradation schedule, gave a ~8:1 lateral:vertical ratio and made
        # channels slither diagonally through Z.
        self.migration_distance = migration_distance_ratio
        self.boundary_reflect = boundary_reflect
        # 'discrete' = Alluvsim-style nlevel aggradation: chelev jumps by one
        # channel depth at each of `nlevel` level boundaries, stamps emit
        # only at boundaries, and the streamline is optionally re-seeded
        # with probability `level_reseed_prob`. Per-Z slices show one crisp
        # meander each, rather than 80 overlaid migration snapshots.
        # 'continuous' = legacy: per-iter tiny chelev increments + stamps
        # every 10 iters. Produces amalgamated channel belt.
        self.aggradation_mode = aggradation_mode
        self.level_reseed_prob = level_reseed_prob
        self.level_jump_ratio = level_jump_ratio
        g = 9.8
        self.g = g
        self.us0 = ((g * Q * I) / (2.0 * b * Cf))**(1.0 / 3.0)
        self.h0 = Q / (2.0 * b * self.us0)
        self.xmin = xmn - 0.5 * xsiz
        self.ymin = ymn - 0.5 * ysiz
        self.xmax = self.xmin + xsiz * nx
        self.ymax = self.ymin + ysiz * ny
        self.station_len = station * self.xmax
        self.x = np.linspace(xmn, self.xmax - xmn, self.nx)
        self.y = np.linspace(ymn, self.ymax - ymn, self.ny)
        self.step = (self.xsiz + self.ysiz) / 2
        self.step0 = (self.xsiz + self.ysiz) * 8
        self.ndis0 = int((((self.xmax - self.xmin) + (self.ymax - self.ymin)) / 2.0) / self.step) * 4
        self.ndis = self.ndis0
        self.incr1 = b / 2
        self.incr2 = b / 4
        self.nthick = int(b / ((xsiz + ysiz) / 2.0))
        self.good = np.zeros([self.nx, self.ny])

    def generate_streamline(self, y0, x0=-1000, k=0.1, s=0.8, h=0.8, m=0):
        k = k * self.step0
        phi = np.arcsin(h)
        b1 = 2.0 * np.exp(-k * h) * np.cos(k * np.cos(phi))
        b2 = -1.0 * np.exp(-2.0 * k * h)
        mm0 = s * np.random.normal(0, 1, size=self.ndis0 + 40)
        mm0 = mm0[20:-20]
        ar = np.array([1, b1, b2])
        theta = scipy.signal.lfilter([1], ar, mm0) + m

        self.cx0 = np.cumsum(self.step0 * np.cos(theta))
        self.cx0 += x0
        self.cy0 = np.cumsum(self.step0 * np.sin(theta))
        self.cy0 += y0
        self.cx0 = np.append([x0], self.cx0)
        self.cy0 = np.append([y0], self.cy0)

        idx0 = np.arange(self.cx0.size)
        if self.cx0.size < 20:
            return 0
        else:
            self.ndis00 = idx0[self.cx0 > (self.xmax)][0] + 4

        self.cx0 = self.cx0[:self.ndis00]
        self.cy0 = self.cy0[:self.ndis00]
        self.length = np.zeros(self.cx0.size)
        self.length[1:] = np.sqrt(
            (self.cx0[1:] - self.cx0[:-1])**2 + (self.cy0[1:] - self.cy0[:-1])**2
        )
        self.length = np.cumsum(self.length)
        self.splx = UnivariateSpline(self.length, self.cx0, k=5, s=0)
        self.sply = UnivariateSpline(self.length, self.cy0, k=5, s=0)

        length = np.linspace(0, self.length[-1], self.cx0.size * 18)
        self.step = length[1] - length[0]
        self.cx = self.splx(length)
        self.cy = self.sply(length)
        self.ndis = self.cx.size

        self.myinit = 10
        if (np.sum(self.cy > self.ymax - self.ny / 2.5 * self.ysiz * (1 - self.chelev / self.nz / self.zsiz))
                + np.sum(self.cy < self.ymin + self.ny / 2.5 * self.ysiz * (1 - self.chelev / self.nz / self.zsiz))) > 0:
            return 0
        else:
            self.x0 = self.cx[0]
            self.y0 = self.cy[0]
            self.x1 = self.cx[-1]
            self.y1 = self.cy[-1]
            return 1

    def cal_curv(self, zzz=0):
        dlength = np.sqrt((self.cx[1:] - self.cx[:-1])**2 + (self.cy[1:] - self.cy[:-1])**2)
        dlength = np.append(0, dlength)
        self.dlength = dlength
        length = np.cumsum(dlength)
        s10 = UnivariateSpline(length, self.cx, k=3, s=0)
        s20 = UnivariateSpline(length, self.cy, k=3, s=0)
        nstep = int((length[-1] - length[0]) / self.step) + 1
        self.length = np.linspace(length[0], length[-1] - 0.1, nstep)

        self.cx = s10(self.length)
        self.cy = s20(self.length)
        zz = 0.2
        self.cx[1:-1] = self.cx[1:-1] * (1 - zz) + zz / 2 * self.cx[:-2] + zz / 2 * self.cx[2:]
        self.cy[1:-1] = self.cy[1:-1] * (1 - zz) + zz / 2 * self.cy[:-2] + zz / 2 * self.cy[2:]

        s1 = UnivariateSpline(self.length, self.cx, k=5, s=4000)
        s2 = UnivariateSpline(self.length, self.cy, k=5, s=4000)

        if zzz == 0:
            self.cx = s10(self.length)
            self.cy = s20(self.length)
        self.splx = s1
        self.sply = s2

        self.chwidth = self.b * np.ones(len(self.length))
        self.ndis = len(self.length)

        vx = s1.derivative(n=1)
        ax = s1.derivative(n=2)
        vy = s2.derivative(n=1)
        ay = s2.derivative(n=2)
        curvature = (
            (vx(self.length) * ay(self.length) - ax(self.length) * vy(self.length))
            / ((vx(self.length)**2 + vy(self.length)**2)**1.5)
        )
        self.curv = curvature

        maxcurvr = max(self.curv) + 0.0001
        maxcurvl = max(-self.curv) + 0.0001
        self.thalweg = self.curv.copy()
        self.thalweg[self.curv >= 0] = 0.5 + self.curv[self.curv >= 0] * 0.25 / maxcurvr
        self.thalweg[self.curv < 0] = 0.5 + self.curv[self.curv < 0] * 0.25 / maxcurvl
        dcsids = (self.curv[1:] - self.curv[:-1]) / (self.length[1:] - self.length[:-1])
        self.dcsids = np.append(dcsids[0], dcsids)

        self.dlength = self.step
        self.vx = vx(self.length)
        self.vy = vy(self.length)

    def find_nearest(self, array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

    def _reseed_streamline(self, s_noise, random_y=False):
        """Draw a new streamline from scratch.

        ``random_y=True`` picks a fresh random Y within the interior band
        (used between aggradation levels in discrete mode).  Otherwise the
        new Y is either a reflection of the current channel's tail across
        the mid-Y (``boundary_reflect=True``, preserves Z-continuity
        after a touch_b event) or simply the mid-Y.
        """
        ymid = 0.5 * (self.ymin + self.ymax)
        if random_y:
            margin = 0.25 * (self.ymax - self.ymin)
            chy = float(np.random.uniform(self.ymin + margin,
                                          self.ymax - margin))
        elif self.boundary_reflect:
            in_grid = ((self.cx > self.xmin) & (self.cx < self.xmax)
                       & (self.cy > self.ymin) & (self.cy < self.ymax))
            tail = self.cy[in_grid][-20:] if in_grid.sum() >= 20 else self.cy
            chy = 2.0 * ymid - float(np.mean(tail))
            margin = 0.25 * (self.ymax - self.ymin)
            chy = float(np.clip(chy, self.ymin + margin, self.ymax - margin))
        else:
            chy = ymid
        success = 0
        attempts = 0
        while success == 0:
            angle = np.random.uniform(-np.pi / 1800, np.pi / 1800)
            success = self.generate_streamline(y0=chy, m=angle, s=s_noise)
            attempts += 1
            if attempts >= 20:
                chy = ymid
        self.chwidth = self.b * np.ones(self.ndis)
        self.maxb = 2 * np.max(self.chwidth)
        self.touch_b = 0

    def _stamp_current_streamline(self, NNN):
        """Paint the current streamline into ``self.facies`` at
        ``self.chelev``.  Assumes ``cal_curv`` has been called so
        ``vx/vy/curv/thalweg`` are current."""
        mygood = ((self.cx > self.xmin) * (self.cx < self.xmax)
                  * (self.cy > self.ymin) * (self.cy < self.ymax))
        mygood = mygood.astype(bool)
        if mygood.sum() < 20:
            return
        cx = self.cx[mygood]
        cy = self.cy[mygood]
        vx_f = self.vx[mygood]
        vy_f = self.vy[mygood]
        curv = self.curv[mygood]
        thalweg = self.thalweg[mygood]
        chwidth = self.chwidth[mygood]
        genchannel(
            self.b, self.xsiz, self.ysiz, self.chelev, self.zsiz,
            self.nx, self.ny, self.nz, cx, cy, self.x, self.y,
            vx_f, vy_f, curv, self.LV_asym, self.lV_height,
            self.ps, self.pavul, self.out, self.totalid, self.facies,
            self.poro, self.poro0, thalweg, chwidth, self.dwratio,
            [1000000000], NNN,
        )

    def _migrate_one_step(self):
        """One Sun-1996 bank-retreat migration step.  Updates ``cx/cy``
        in place and populates ``self.idxx`` with the keep-mask from
        ``make_cutoff``.  Sets ``self.touch_b=1`` if the streamline is
        approaching a Y-boundary.  Returns 0 if the streamline is too
        short to continue, 1 otherwise."""
        if self.cx.size < 20:
            return 0
        self.cal_curv()
        self.usbmat = np.zeros(self.ndis)
        for idis in np.arange(1, self.ndis):
            dx = self.splx.integral(self.length[idis - 1], self.length[idis])
            dy = self.sply.integral(self.length[idis - 1], self.length[idis])
            dlength = np.sqrt(dx**2 + dy**2)
            self.usbmat[idis] = (
                self.b / (self.us0 / dlength + 2 * (self.us0 / self.h0) * self.Cf)
                * (-self.us0**2 * self.dcsids[idis]
                   + self.Cf * self.curv[idis] * (self.us0**4 / self.g / self.h0**2
                                                  + self.A * self.us0**2 / self.h0)
                   + self.us0 / dlength * self.usbmat[idis - 1] / self.b)
            )
        self.usbmat = np.abs(self.usbmat)

        tmigrate = self.migration_distance * (self.xsiz + self.ysiz)
        vt = np.sqrt(self.vx**2 + self.vy**2)
        damp = np.ones(self.cx.size)
        dist = np.arange(self.cx.size)
        damp = 2 - 2 / (1 + np.exp(-np.abs(dist - dist.mean()) / 100))
        self.usbmat_vx = (np.sign(self.curv) * self.vy / vt * self.usbmat * damp
                          + 100 * self.us0 * self.vx / vt)
        self.usbmat_vy = (-np.sign(self.curv) * self.vx / vt * self.usbmat * damp
                          + 100 * self.us0 * self.vy / vt)
        self.usbmat_t = np.sqrt(self.usbmat_vx**2 + self.usbmat_vy**2)
        self.maxmigrate = np.max(self.usbmat_t)
        self.E = tmigrate / self.maxmigrate

        self.cy[:21] = self.cy[16]
        self.cx[20:] = self.cx[20:] + self.usbmat_vx[20:] * self.E
        self.cy[20:] = self.cy[20:] + self.usbmat_vy[20:] * self.E

        cut_dist = int(1 * self.ndis / 2)
        thresh = self.b * 2.5
        idxx = np.ones(self.ndis)
        make_cutoff(self.step, self.ndis, self.dlength, thresh, cut_dist,
                    self.cx, self.cy, idxx, self.totalid)

        if (np.sum(self.cy > self.ymax - self.ny / 5 * self.ysiz * (1 - self.chelev / self.nz / self.zsiz))
                + np.sum(self.cy < self.ymin + self.ny / 5 * self.ysiz * (1 - self.chelev / self.nz / self.zsiz))) > 0:
            if self.totalid > 10:
                self.touch_b = 1

        idxx[self.cx < self.x0] = 0
        idxx[self.cx > self.x1] = 0
        self.idxx = idxx
        return 1

    def _trim_streamline(self):
        """Apply the current idxx cutoff mask and restore pinned endpoints.
        Returns 0 if too few nodes remain, 1 otherwise."""
        self.cx = self.cx[self.idxx.astype(bool)]
        self.cy = self.cy[self.idxx.astype(bool)]
        if self.cx.size < 20:
            return 0
        self.cx[0] = self.x0
        self.cy[0] = self.y0
        self.cx[-1] = self.x1
        self.cy[-1] = self.y1
        return 1

    def simulation(self, nchannel=10):
        self.itime = 0
        self.facies = np.zeros((self.nx, self.ny, self.nz))
        totalid = 0
        self.totalid = totalid
        success = 0
        self.out = 0
        self.cz = int(self.dwratio * self.b / self.zsiz)
        self.chelev = (self.cz + 1) * self.zsiz

        # Quadratic mapping gives smooth user control:
        # meander_scale=0 → straight, 1 → moderate, 2 → legacy default (s=0.8)
        s_noise = 0.2 * self.meander_scale ** 2

        while success == 0:
            chy = np.random.uniform(
                self.ymin + (self.ymax - self.ymin) / 2,
                self.ymin + (self.ymax - self.ymin) / 2,
            )
            angle = np.random.uniform(-np.pi / 1800, np.pi / 1800)
            success = self.generate_streamline(y0=chy, m=angle, s=s_noise)
        idxx = np.ones(self.cx.size)
        self.chelev = (self.cz + 1) * self.zsiz
        self.touch_b = 0
        ntg = 0
        self.myinit = 10
        self.mybot = 0

        NNN = nchannel * 10
        if self.aggradation_mode == 'discrete':
            return self._simulate_discrete(nchannel, NNN, s_noise)

        for ddd in range(NNN):
            self.myinit -= 1
            self.ps = 1
            self.out = 0

            self.chelev = self.chelev + self.aggrad[int((self.totalid - 1) / (NNN / len(self.aggrad)))]

            totalid += 1
            self.totalid = totalid

            if self.touch_b == 1:
                self.out = 1
                self.cal_curv()
                genchannel(
                    self.b, self.xsiz, self.ysiz, self.chelev, self.zsiz,
                    self.nx, self.ny, self.nz, self.cx, self.cy, self.x, self.y,
                    self.vx, self.vy, self.curv, self.LV_asym, self.lV_height,
                    self.ps, self.pavul, self.out, self.totalid, self.facies,
                    self.poro, self.poro0, self.thalweg, self.chwidth,
                    self.dwratio, [10000000000], NNN,
                )
                self.out = 0

                ymid = 0.5 * (self.ymin + self.ymax)
                if self.boundary_reflect:
                    # Reflect the last in-grid segment across ymid instead of
                    # restarting at mid-Y. Preserves Z-continuity of the
                    # channel belt across boundary-touch events.
                    in_grid = ((self.cx > self.xmin) & (self.cx < self.xmax)
                               & (self.cy > self.ymin) & (self.cy < self.ymax))
                    tail = self.cy[in_grid][-20:] if in_grid.sum() >= 20 else self.cy
                    chy = 2.0 * ymid - float(np.mean(tail))
                    margin = 0.25 * (self.ymax - self.ymin)
                    chy = float(np.clip(chy, self.ymin + margin, self.ymax - margin))
                else:
                    chy = ymid
                success = 0
                attempts = 0
                while success == 0:
                    angle = np.random.uniform(-np.pi / 1800, np.pi / 1800)
                    success = self.generate_streamline(y0=chy, m=angle, s=s_noise)
                    attempts += 1
                    if attempts >= 20:
                        chy = ymid
                self.chwidth = self.b * np.ones(self.ndis)
                self.maxb = 2 * np.max(self.chwidth)
                self.touch_b = 0
                continue

            if self.cx.size < 20:
                return 0

            self.cal_curv()

            # Calculate near bank velocity
            self.usbmat = np.zeros(self.ndis)
            for idis in np.arange(1, self.ndis):
                dx = self.splx.integral(self.length[idis - 1], self.length[idis])
                dy = self.sply.integral(self.length[idis - 1], self.length[idis])
                dlength = np.sqrt(dx**2 + dy**2)
                self.usbmat[idis] = (
                    self.b / (self.us0 / dlength + 2 * (self.us0 / self.h0) * self.Cf)
                    * (-self.us0**2 * self.dcsids[idis]
                       + self.Cf * self.curv[idis] * (self.us0**4 / self.g / self.h0**2
                                                      + self.A * self.us0**2 / self.h0)
                       + self.us0 / dlength * self.usbmat[idis - 1] / self.b)
                )
            self.usbmat = np.abs(self.usbmat)

            tmigrate = self.migration_distance * (self.xsiz + self.ysiz)
            vt = np.sqrt(self.vx**2 + self.vy**2)

            damp = np.ones(self.cx.size)
            dist = np.arange(self.cx.size)
            damp = 2 - 2 / (1 + np.exp(-np.abs(dist - dist.mean()) / 100))

            self.usbmat_vx = (np.sign(self.curv) * self.vy / vt * self.usbmat * damp
                              + 100 * self.us0 * self.vx / vt)
            self.usbmat_vy = (-np.sign(self.curv) * self.vx / vt * self.usbmat * damp
                              + 100 * self.us0 * self.vy / vt)
            self.usbmat_t = np.sqrt(self.usbmat_vx**2 + self.usbmat_vy**2)
            self.maxmigrate = np.max(self.usbmat_t)
            self.E = tmigrate / self.maxmigrate

            self.cy[:21] = self.cy[16]
            self.cx[20:] = self.cx[20:] + self.usbmat_vx[20:] * self.E
            self.cy[20:] = self.cy[20:] + self.usbmat_vy[20:] * self.E

            cut_dist = int(1 * self.ndis / 2)
            thresh = self.b * 2.5
            idxx = np.ones(self.ndis)
            make_cutoff(self.step, self.ndis, self.dlength, thresh, cut_dist,
                        self.cx, self.cy, idxx, self.totalid)

            if (np.sum(self.cy > self.ymax - self.ny / 5 * self.ysiz * (1 - self.chelev / self.nz / self.zsiz))
                    + np.sum(self.cy < self.ymin + self.ny / 5 * self.ysiz * (1 - self.chelev / self.nz / self.zsiz))) > 0:
                if self.totalid > 10:
                    self.touch_b = 1

            idxx[self.cx < self.x0] = 0
            idxx[self.cx > self.x1] = 0

            self.out = 0
            if self.totalid % 10 == 9:
                mygood = ((self.cx > self.xmin) * (self.cx < self.xmax)
                          * (self.cy > self.ymin) * (self.cy < self.ymax))
                mygood = mygood.astype(bool)
                cx = self.cx[mygood]
                cy = self.cy[mygood]
                vx_f = self.vx[mygood]
                vy_f = self.vy[mygood]
                curv = self.curv[mygood]
                thalweg = self.thalweg[mygood]
                chwidth = self.chwidth[mygood]
                genchannel(
                    self.b, self.xsiz, self.ysiz, self.chelev, self.zsiz,
                    self.nx, self.ny, self.nz, cx, cy, self.x, self.y,
                    vx_f, vy_f, curv, self.LV_asym, self.lV_height,
                    self.ps, self.pavul, self.out, self.totalid, self.facies,
                    self.poro, self.poro0, thalweg, chwidth, self.dwratio,
                    [1000000000], NNN,
                )

            self.cx = self.cx[idxx.astype(bool)]
            self.cy = self.cy[idxx.astype(bool)]
            if self.cx.size < 20:
                return 0
            self.cx[0] = self.x0
            self.cy[0] = self.y0
            self.cx[-1] = self.x1
            self.cy[-1] = self.y1
            self.idxx = idxx

        self.cal_curv()

    def _simulate_discrete(self, nchannel, NNN, s_noise):
        """Alluvsim-style nlevel aggradation.

        The streamline migrates for ``iters_per_level`` iterations at a
        fixed ``chelev``; a single snapshot is stamped at the end of the
        level; ``chelev`` jumps by one channel depth; with probability
        ``level_reseed_prob`` the streamline is re-drawn from a fresh
        random Y for the next level.  Cf. Alluvsim's ``streamsim.for``
        main loop + ``pv_shoestring`` preset (``nlevel=5`` over
        ``ntime=120`` iterations, ``probAvulOutside+probAvulInside≈0.15``).

        Per-Z slices each show one crisp meander rather than ~80 overlaid
        migration snapshots piled into ~12 Z cells.
        """
        totalid = self.totalid
        nlevel = max(1, self.nlevel)
        iters_per_level = max(1, NNN // nlevel)
        # jump = level_jump_ratio × channel depth.  ratio=1.0 stacks
        # stamps with no vertical overlap (Alluvsim pv_shoestring); ratio
        # <1 gives increasing overlap → a single continuous channel belt
        # when combined with level_reseed_prob=0.
        jump = self.level_jump_ratio * self.cz * self.zsiz

        for ilevel in range(nlevel):
            for inner in range(iters_per_level):
                self.myinit -= 1
                self.ps = 1
                self.out = 0
                totalid += 1
                self.totalid = totalid

                if self.touch_b == 1:
                    self.out = 1
                    self.cal_curv()
                    self._stamp_current_streamline(NNN)
                    self.out = 0
                    self._reseed_streamline(s_noise)
                    continue

                if self._migrate_one_step() == 0:
                    break
                if self._trim_streamline() == 0:
                    break

            self.cal_curv()
            self._stamp_current_streamline(NNN)

            self.chelev = self.chelev + jump

            if ilevel < nlevel - 1 and np.random.uniform() < self.level_reseed_prob:
                self._reseed_streamline(s_noise, random_y=True)

        self.cal_curv()
