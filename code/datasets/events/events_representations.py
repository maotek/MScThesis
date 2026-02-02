import torch
from abc import ABC, abstractmethod


class EventRepresentation(ABC):

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    @abstractmethod
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        pass



class VoxelGrid(EventRepresentation):
    def __init__(self, channels: int, height: int, width: int, normalize: bool):
        super().__init__(height, width)
        self.voxel_grid = torch.zeros((channels, height, width), dtype=torch.float, requires_grad=False)
        self.channels = channels
        self.normalize = normalize
        self.height = height
        self.width = width

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        # C, H, W = self.voxel_grid.shape
        # with torch.no_grad():
        #     self.voxel_grid = self.voxel_grid.to(pol.device)
        #     voxel_grid = self.voxel_grid.clone()

        #     if x.shape[0] == 0:
        #         return voxel_grid

        #     t_norm = time
        #     t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0]) # This normalizes t between 0 and C-1

        #     x0 = x.int() # Let's make the x an integer
        #     y0 = y.int() # Let's make the y an integer
        #     t0 = t_norm.int() # Let's make the normalized time an integer

        #     value = 2*pol-1 # Let's make pol from in [0; 1] to in [-1; 1]

        #     for xlim in [x0,x0+1]:
        #         for ylim in [y0,y0+1]:
        #             for tlim in [t0,t0+1]:

        #                 mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.channels)
        #                 interp_weights = value * (1 - (xlim-x).abs()) * (1 - (ylim-y).abs()) * (1 - (tlim - t_norm).abs())

        #                 index = H * W * tlim.long() + \
        #                         W * ylim.long() + \
        #                         xlim.long()

        #                 voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

        #     if self.normalize:
        #         voxel_grid = self.normalize_fn(voxel_grid)

        C, H, W = self.voxel_grid.shape
        device = x.device

        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(device)
            voxel_grid = self.voxel_grid.clone()

            if x.numel() == 0:
                return voxel_grid

            # normalize time → [0, C-1]
            t_norm = (C - 1) * (time - time[0]) / (time[-1] - time[0] + 1e-12)

            x = x.long()
            y = y.long()
            t = t_norm.long()
            val = 2 * pol.float() - 1.0  # [0,1] → [-1,1]

            # valid mask
            valid = (x >= 0) & (x < W) & \
                    (y >= 0) & (y < H) & \
                    (t >= 0) & (t < C)

            x, y, t, val = x[valid], y[valid], t[valid], val[valid]

            # flat voxel index
            idx = t * (H * W) + y * W + x

            flat = voxel_grid.view(-1)

            # deterministic accumulation
            flat.scatter_add_(0, idx, val)

            if self.normalize:
                voxel_grid = self.normalize_fn(voxel_grid)

            # print(voxel_grid.device)
            # print(x.sum(), y.sum(), pol.sum(), time.sum())
            # print(voxel_grid.sum())

        return voxel_grid
    
    def normalize_fn_enrico(self, voxel_grid):
        mask = torch.nonzero(voxel_grid, as_tuple=True)
        if mask[0].size()[0] > 0:
            mean = voxel_grid[mask].mean()
            std = voxel_grid[mask].std()
            if std > 0:
                voxel_grid[mask] = (voxel_grid[mask] - mean) / std
            else:
                voxel_grid[mask] = voxel_grid[mask] - mean
        
        return voxel_grid
    
    def normalize_fn(self, voxel_grid):
        nonzero_ev = (voxel_grid != 0)
        num_nonzeros = nonzero_ev.sum()
        if num_nonzeros > 0:
            mean = torch.sum(voxel_grid, dtype=torch.float32) / num_nonzeros  # force torch.float32 to prevent overflows when using 16-bit precision
            stddev = torch.sqrt(torch.sum(voxel_grid ** 2, dtype=torch.float32) / num_nonzeros - mean ** 2)
            mask = nonzero_ev.type_as(voxel_grid)
            voxel_grid = mask * (voxel_grid - mean) / stddev

        return voxel_grid


class ETNetVoxelGrid(EventRepresentation):
    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        combined_voxel_channels: bool = True,
        temporal_bilinear: bool = True,
    ):
        super().__init__(height, width)
        self.channels = channels
        self.combined_voxel_channels = combined_voxel_channels
        self.temporal_bilinear = temporal_bilinear
        self.height = height
        self.width = width
        if combined_voxel_channels:
            self.num_bins = channels
        else:
            if channels % 2 != 0:
                raise ValueError("channels must be even when combined_voxel_channels=False")
            self.num_bins = channels // 2

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        device = x.device
        if x.numel() == 0:
            return torch.zeros((self.channels, self.height, self.width), dtype=torch.float32, device=device)

        with torch.no_grad():
            # Follows ET-Net voxelization in models/etnet/events_contrast_maximization/utils/event_utils.py
            xs = x.to(device)
            ys = y.to(device)
            ts = time.to(device)
            ps = pol.to(device).float()
            if ps.min() >= 0 and ps.max() <= 1:
                ps = ps * 2.0 - 1.0

            dt = ts[-1] - ts[0]
            if dt == 0:
                dt = torch.tensor(1.0, device=device)
            t_norm = (ts - ts[0]) / dt * (self.num_bins - 1)

            valid = (
                (xs >= 0)
                & (xs < self.width)
                & (ys >= 0)
                & (ys < self.height)
            )
            if not torch.all(valid):
                xs = xs[valid]
                ys = ys[valid]
                ps = ps[valid]
                t_norm = t_norm[valid]

            if xs.numel() == 0:
                return torch.zeros((self.channels, self.height, self.width), dtype=torch.float32, device=device)

            voxel_bins = []
            zeros = torch.zeros_like(t_norm)
            xs_long = xs.long()
            ys_long = ys.long()
            for bi in range(self.num_bins):
                if self.temporal_bilinear:
                    weights = ps * torch.max(zeros, 1.0 - torch.abs(t_norm - bi))
                    img = torch.zeros((self.height, self.width), dtype=torch.float32, device=device)
                    img.index_put_((ys_long, xs_long), weights, accumulate=True)
                else:
                    img = torch.zeros((self.height, self.width), dtype=torch.float32, device=device)
                    img.index_put_((ys_long, xs_long), ps, accumulate=True)
                voxel_bins.append(img)

            voxel_grid = torch.stack(voxel_bins, dim=0)

            if not self.combined_voxel_channels:
                pos = torch.clamp(voxel_grid, min=0.0)
                neg = torch.clamp(-voxel_grid, min=0.0)
                voxel_grid = torch.cat([pos, neg], dim=0)

        return voxel_grid


class E2vidVoxelGrid(EventRepresentation):
    def __init__(self, channels: int, height: int, width: int):
        super().__init__(height, width)
        self.channels = channels
        self.height = height
        self.width = width

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        device = x.device
        if x.numel() == 0:
            return torch.zeros((self.channels, self.height, self.width), dtype=torch.float32, device=device)

        with torch.no_grad():
            # Follows E2VID voxel grid in models/rpg_e2vid/utils/inference_utils.py
            voxel_grid = torch.zeros(
                self.channels, self.height, self.width, dtype=torch.float32, device=device
            ).flatten()

            first_stamp = time[0]
            last_stamp = time[-1]
            delta_t = last_stamp - first_stamp
            if delta_t == 0:
                delta_t = torch.tensor(1.0, device=device)

            ts = (self.channels - 1) * (time - first_stamp) / delta_t
            xs = x.long()
            ys = y.long()
            pols = pol.float()
            pols = torch.where(pols == 0, torch.tensor(-1.0, device=device), pols)

            tis = torch.floor(ts)
            tis_long = tis.long()
            dts = ts - tis
            vals_left = pols * (1.0 - dts)
            vals_right = pols * dts

            valid = (
                (tis < self.channels)
                & (tis >= 0)
                & (xs >= 0)
                & (xs < self.width)
                & (ys >= 0)
                & (ys < self.height)
            )
            idx_left = xs[valid] + ys[valid] * self.width + tis_long[valid] * self.width * self.height
            voxel_grid.index_add_(0, idx_left, vals_left[valid])

            valid = (
                ((tis + 1) < self.channels)
                & (tis >= 0)
                & (xs >= 0)
                & (xs < self.width)
                & (ys >= 0)
                & (ys < self.height)
            )
            idx_right = xs[valid] + ys[valid] * self.width + (tis_long[valid] + 1) * self.width * self.height
            voxel_grid.index_add_(0, idx_right, vals_right[valid])

            voxel_grid = voxel_grid.view(self.channels, self.height, self.width)

            # Normalize nonzero events to zero-mean/unit-std (matches E2VID preprocessing)
            nonzero_ev = voxel_grid != 0
            num_nonzeros = nonzero_ev.sum()
            if num_nonzeros > 0:
                mean = torch.sum(voxel_grid, dtype=torch.float32) / num_nonzeros
                stddev = torch.sqrt(torch.sum(voxel_grid ** 2, dtype=torch.float32) / num_nonzeros - mean ** 2)
                mask = nonzero_ev.type_as(voxel_grid)
                voxel_grid = mask * (voxel_grid - mean) / stddev

        return voxel_grid


class Histogram(EventRepresentation):

    def __init__(self, height: int, width: int, remove_int_artifact: bool):
        super().__init__(height, width)
        self.height = height
        self.width = width
        self.remove_int_artifact = remove_int_artifact

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        mask = (x < self.width) & (x >= 0) & (y < self.height) & (y >= 0)

        x = x[mask]
        y = y[mask]
        pol = pol[mask]

        histo = torch.zeros((2, self.height, self.width), dtype=torch.float, requires_grad=False)

        if x.shape[0] == 0:
            return histo

        with (torch.no_grad()):
            pol = pol.int() # Let's make the polarity an integer

            x0 = x.int()  # Let's make the x an integer
            y0 = y.int()  # Let's make the y an integer

            if self.remove_int_artifact:
                for x_lim in [x0, x0 + 1]:
                    for y_lim in [y0, y0 + 1]:
                        interp_weights = ((1 - (x_lim - x).abs()) + (1 - (y_lim - y).abs())) / 4
                        index = pol*self.height*self.width + self.width * y_lim.long() + x_lim.long()
                        mask = index < self.height*self.width*2
                        histo.put_(index[mask], interp_weights[mask], accumulate=True)
            else:
                index = pol * self.height * self.width + self.width * y0.long() + x0.long()
                histo.put_(index, torch.ones_like(index, dtype=histo.dtype), accumulate=True)
            # CLIP:
            histo[histo > 200] = 200
        return histo



class Tencode(EventRepresentation):

    def __init__(self, height: int, width: int, normalize: bool, white_frame: bool = False):
        super().__init__(height, width)
        self.height = height
        self.width = width
        self.normalize = normalize
        self.white_frame = white_frame

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        # if self.white_frame:
        #     tencode = torch.full((3, self.height, self.width), 255.0, dtype=torch.float, requires_grad=False)
        # else:
        #     tencode = torch.zeros((3, self.height, self.width), dtype=torch.float, requires_grad=False)

        # if x.shape[0] == 0:
        #     return tencode
        # with (torch.no_grad()):
        #     pol = pol.int() # Let's make the polarity an integer {0,1}

        #     t_norm = time
        #     t_norm = (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

        #     index_red = (0 * self.width * self.height) + (y.long() * self.width) + x.long()
        #     index_green = (1 * self.width * self.height) + (y.long() * self.width) + x.long()
        #     index_blue = (2 * self.width * self.height) + (y.long() * self.width) + x.long()

        #     mask_red = (x < self.width) & (x >= 0) & (y < self.height) & (y >= 0) & (index_red >= 0) \
        #         & (index_red < 3*self.height*self.width)
        #     mask_green = (x < self.width) & (x >= 0) & (y < self.height) & (y >= 0) & (index_green >= 0) \
        #         & (index_green < 3*self.height*self.width)
        #     mask_blue = (x < self.width) & (x >= 0) & (y < self.height) & (y >= 0) & (index_blue >= 0) \
        #         & (index_blue < 3*self.height*self.width)
            
        #     tencode.put_(index_red[mask_red], 255.0*pol[mask_red], accumulate=False)
        #     tencode.put_(index_green[mask_green], 255.0*(1-t_norm[mask_green]), accumulate=False)
        #     tencode.put_(index_blue[mask_blue], 255.0*(1-pol[mask_blue]), accumulate=False)
        #     print(tencode.device)
        #     print(x.sum(), y.sum(), pol.sum(), time.sum())
        #     print(tencode.sum())

        #     if self.normalize:
        #         tencode = tencode / 255.0

        H, W = self.height, self.width
        device = x.device

        # init output
        if self.white_frame:
            tencode = torch.full((3, H, W), 255.0, device=device)
        else:
            tencode = torch.zeros((3, H, W), device=device)

        if x.numel() == 0:
            return tencode / 255.0 if self.normalize else tencode

        with torch.no_grad():
            x = x.long()
            y = y.long()
            pol = pol.int()

            # valid events
            valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            x, y, pol, time = x[valid], y[valid], pol[valid], time[valid]

            if x.numel() == 0:
                return tencode / 255.0 if self.normalize else tencode

            # normalize time
            t_norm = (time - time[0]) / (time[-1] - time[0])

            # pixel index
            idx = y * W + x
            HW = H * W

            # store max time per pixel
            max_time = torch.full((HW,), -1.0, device=device)
            max_time.scatter_reduce_(0, idx, time, reduce="amax")

            # keep only "last" events
            keep = time == max_time[idx]

            idx = idx[keep]
            pol = pol[keep]
            t_norm = t_norm[keep]

            flat = tencode.view(3, -1)

            flat[0, idx] = 255.0 * pol.float()
            flat[1, idx] = 255.0 * (1.0 - t_norm)
            flat[2, idx] = 255.0 * (1.0 - pol.float())

            if self.normalize:
                tencode /= 255.0
            
            # print(tencode.device)
            # print(x.sum(), y.sum(), pol.sum(), time.sum())
            # print(tencode.sum())

        return tencode



class TencodePixelCount(EventRepresentation):
    
    def __init__(self, height: int, width: int, normalize: bool, white_frame: bool = False):
        super().__init__(height, width)
        self.height = height
        self.width = width
        self.normalize = normalize
        self.white_frame = white_frame

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        H, W = self.height, self.width
        device = x.device

        base_value = 255.0 if self.white_frame else 0.0
        representation = torch.full((3, H, W), base_value, device=device)

        if x.numel() == 0:
            return representation / 255.0 if self.normalize else representation

        with torch.no_grad():
            x = x.long()
            y = y.long()
            pol = pol.int()

            valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            x, y, pol, time = x[valid], y[valid], pol[valid], time[valid]

            if x.numel() == 0:
                return representation / 255.0 if self.normalize else representation

            denom = torch.clamp(time[-1] - time[0], min=1e-12)
            t_norm = (time - time[0]) / denom

            idx = y * W + x
            HW = H * W

            counts = torch.zeros((HW,), device=device)
            counts.scatter_add_(0, idx, torch.ones_like(time, dtype=torch.float, device=device))

            max_time = torch.full((HW,), -1.0, device=device)
            max_time.scatter_reduce_(0, idx, time, reduce="amax")

            keep = time == max_time[idx]
            idx_last = idx[keep]
            pol_last = pol[keep]
            t_norm_last = t_norm[keep]

            flat = representation.view(3, -1)
            red_values = torch.where(pol_last == 1,
                                     torch.tensor(125.0, device=device),
                                     torch.tensor(255.0, device=device))
            flat[2, idx_last] = red_values
            flat[1, idx_last] = 255.0 * (1.0 - t_norm_last)

            count_mask = counts > 0
            if count_mask.any():
                nz_counts = counts[count_mask]
                cmin = nz_counts.min()
                cmax = nz_counts.max()
                if cmax > cmin:
                    scaled = 255.0 * (nz_counts - cmin) / (cmax - cmin)
                else:
                    scaled = torch.full_like(nz_counts, 255.0)
                flat[0, count_mask] = scaled
            # flat[0, :] = 0
            # flat[1, :] = 0
            if self.normalize:
                representation = representation / 255.0

        return representation


class E2DepthVoxelGrid(EventRepresentation):
    """
    Voxel grid representation used by E2Depth with bilinear temporal interpolation.
    Based on events_to_voxel_grid_pytorch from models/rpg_e2depth/utils/event_tensor_utils.py
    """
    def __init__(self, channels: int, height: int, width: int):
        super().__init__(height, width)
        self.channels = channels
        self.height = height
        self.width = width

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        device = x.device
        if x.numel() == 0:
            return torch.zeros((self.channels, self.height, self.width), dtype=torch.float32, device=device)

        with torch.no_grad():
            # Flatten voxel grid for index-based operations
            voxel_grid = torch.zeros(
                self.channels, self.height, self.width, dtype=torch.float32, device=device
            ).flatten()

            # Normalize timestamps to [0, num_bins-1]
            first_stamp = time[0]
            last_stamp = time[-1]
            delta_t = last_stamp - first_stamp
            if delta_t == 0:
                delta_t = torch.tensor(1.0, device=device)

            ts = (self.channels - 1) * (time - first_stamp) / delta_t
            xs = x.long()
            ys = y.long()
            pols = pol.float()
            # Convert polarity from [0,1] to [-1,1] if needed
            pols = torch.where(pols == 0, torch.tensor(-1.0, device=device), pols)

            # Bilinear temporal interpolation
            tis = torch.floor(ts)
            tis_long = tis.long()
            dts = ts - tis
            vals_left = pols * (1.0 - dts)
            vals_right = pols * dts

            # Add events to left bin
            valid = (
                (tis < self.channels)
                & (tis >= 0)
                & (xs >= 0)
                & (xs < self.width)
                & (ys >= 0)
                & (ys < self.height)
            )
            idx_left = xs[valid] + ys[valid] * self.width + tis_long[valid] * self.width * self.height
            voxel_grid.index_add_(0, idx_left, vals_left[valid])

            # Add events to right bin
            valid = (
                ((tis + 1) < self.channels)
                & (tis >= 0)
                & (xs >= 0)
                & (xs < self.width)
                & (ys >= 0)
                & (ys < self.height)
            )
            idx_right = xs[valid] + ys[valid] * self.width + (tis_long[valid] + 1) * self.width * self.height
            voxel_grid.index_add_(0, idx_right, vals_right[valid])

            voxel_grid = voxel_grid.view(self.channels, self.height, self.width)

        return voxel_grid
