"""MosaicML Streaming dataset loader for ZipDepth.

Enables high-throughput chunked shard streaming (.mds) across multi-domain
datasets scattered across independent local or mounted directories (e.g. Kaggle datasets).
"""

import os
import glob
import json
import yaml
import numpy as np
import cv2
cv2.setNumThreads(0)
import torch
from torch.utils.data import DataLoader

try:
    from streaming import StreamingDataset, Stream
    from streaming.base.dataset import _ShardState
    STREAMING_AVAILABLE = True
except ImportError:
    StreamingDataset = object
    Stream = None
    _ShardState = None
    STREAMING_AVAILABLE = False


def _patch_streaming_hf_downloader():
    """Patches MosaicML's HFDownloader to stream directly from Hugging Face CloudFront CDN.

    Bypasses HF's /api/datasets/.../xet-read-token REST API (1,000 req / 5 min limit) and
    explicitly attaches the user's HF_TOKEN for authenticated quota (up to 5,000 req / 5 min).
    """
    try:
        from streaming.base.storage.download import HFDownloader
        import requests
        import random
        import time

        def _direct_cdn_download(self, remote, local, timeout):
            parts = remote.split('/')
            if len(parts) >= 6 and parts[0] == 'hf:' and parts[2] == 'datasets':
                repo_org = parts[3]
                repo_name = parts[4]
                subpath = '/'.join(parts[5:])
                cdn_url = f"https://huggingface.co/datasets/{repo_org}/{repo_name}/resolve/main/{subpath}"
            else:
                from huggingface_hub import hf_hub_download
                _, _, _, repo_org, repo_name, path = remote.split('/', 5)
                local_dirname = os.path.dirname(local)
                hf_hub_download(repo_id=f'{repo_org}/{repo_name}',
                                filename=path,
                                repo_type='dataset',
                                local_dir=local_dirname)
                downloaded_name = os.path.join(local_dirname, path)
                os.rename(downloaded_name, local)
                return

            local_dirname = os.path.dirname(local)
            os.makedirs(local_dirname, exist_ok=True)

            if os.path.isfile(local) and os.path.getsize(local) > 0:
                return

            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if not token:
                try:
                    from kaggle_secrets import UserSecretsClient
                    token = UserSecretsClient().get_secret("HF_TOKEN")
                except Exception:
                    pass

            headers = {"User-Agent": "ZipDepth-MosaicML-Streaming/1.0"}
            if token and token.strip():
                headers["Authorization"] = f"Bearer {token.strip()}"

            tmp_file = f"{local}.tmp.{os.getpid()}"
            max_retries = 5
            req_timeout = timeout if timeout and timeout > 0 else 180

            for attempt in range(max_retries):
                try:
                    with requests.get(cdn_url, headers=headers, stream=True, timeout=(15, req_timeout)) as resp:
                        if resp.status_code == 429:
                            retry_after = resp.headers.get("Retry-After")
                            wait_time = float(retry_after) if retry_after else (2 ** attempt * 2 + random.uniform(1.0, 2.5))
                            time.sleep(wait_time)
                            continue

                        resp.raise_for_status()

                        with open(tmp_file, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)

                        os.replace(tmp_file, local)
                        return
                except Exception as e:
                    if os.path.exists(tmp_file):
                        try:
                            os.remove(tmp_file)
                        except Exception:
                            pass
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(2 ** attempt * 1.5 + random.uniform(0.5, 1.5))

        HFDownloader._download_file_impl = _direct_cdn_download
    except Exception:
        pass


if STREAMING_AVAILABLE:
    _patch_streaming_hf_downloader()

try:
    from turbojpeg import TurboJPEG, TJPF_RGB
    _TURBO_JPEG = TurboJPEG()
except Exception:
    _TURBO_JPEG = None

from zipdepth.data.transforms import get_train_transforms, AlbumentationsWrapper


def _decode_rgb(raw_data):
    """Decode raw RGB bytes (JPEG/PNG) to an RGB numpy array uint8 [H, W, 3]."""
    if isinstance(raw_data, np.ndarray):
        if raw_data.ndim == 3 and raw_data.shape[2] == 3:
            return raw_data
        raw_bytes = raw_data.tobytes()
    elif isinstance(raw_data, bytes):
        raw_bytes = raw_data
    else:
        raise ValueError(f"Unsupported RGB data type: {type(raw_data)}")

    if _TURBO_JPEG is not None:
        try:
            return _TURBO_JPEG.decode(raw_bytes, pixel_format=TJPF_RGB)
        except Exception:
            pass

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode RGB image buffer.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _decode_depth(raw_data):
    """Decode raw depth bytes (16-bit PNG) to an uint16 numpy array [H, W]."""
    if isinstance(raw_data, np.ndarray):
        if raw_data.ndim == 2:
            return raw_data
        elif raw_data.ndim == 3:
            return raw_data.squeeze()
        raw_bytes = raw_data.tobytes()
    elif isinstance(raw_data, bytes):
        raw_bytes = raw_data
    else:
        raise ValueError(f"Unsupported Depth data type: {type(raw_data)}")

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError("Failed to decode Depth map buffer.")
    if depth.ndim == 3:
        depth = depth.squeeze()
    return depth


def build_streams_from_sources(sources, default_proportions=None):
    """
    Constructs a list of MosaicML Stream objects from various input specifications.

    Args:
        sources: Can be:
            - A list of directories or glob patterns (e.g. ['/kaggle/input/ade20k', '/kaggle/input/hrwsi'])
            - A comma-separated string (e.g. '/kaggle/input/ade20k,/kaggle/input/hrwsi')
            - A path to a YAML/JSON configuration file
            - A list of dicts with Stream kwargs (e.g. [{'local': '...', 'proportion': 0.5}])
        default_proportions: Optional list of float weights. If None, streams are weighted equally.

    Returns:
        list of Stream objects
    """
    if not STREAMING_AVAILABLE:
        raise ImportError(
            "mosaicml-streaming is required to use MDS datasets. "
            "Install it via: pip install mosaicml-streaming"
        )

    # 1. Check if sources is a file path to YAML/JSON
    if isinstance(sources, str) and (sources.endswith('.yaml') or sources.endswith('.yml') or sources.endswith('.json')):
        if not os.path.exists(sources):
            raise FileNotFoundError(f"Streams configuration file not found: {sources}")
        with open(sources, 'r') as f:
            if sources.endswith('.json'):
                cfg = json.load(f)
            else:
                cfg = yaml.safe_load(f)

        stream_specs = cfg.get('streams', cfg)
        if not isinstance(stream_specs, list):
            raise ValueError(f"Streams config must define a list of stream definitions. Got: {type(stream_specs)}")
        
        streams = []
        for s in stream_specs:
            if isinstance(s, dict):
                streams.append(Stream(**s))
            elif isinstance(s, str):
                streams.append(Stream(local=s))
        return streams

    # 2. If comma-separated string or glob pattern
    if isinstance(sources, str):
        if ',' in sources:
            dir_list = [d.strip() for d in sources.split(',') if d.strip()]
        else:
            expanded = glob.glob(sources)
            dir_list = expanded if expanded else [sources]
    elif isinstance(sources, (list, tuple)):
        dir_list = list(sources)
    else:
        raise ValueError(f"Unsupported sources format: {type(sources)}")

    # Resolve any list of dicts vs list of directory paths
    resolved_dirs = []
    for item in dir_list:
        if isinstance(item, dict):
            # Already a stream dict specification
            continue
        # Check if item is a glob
        if any(c in item for c in ['*', '?', '[']):
            matched = sorted(glob.glob(item))
            resolved_dirs.extend([m for m in matched if os.path.isdir(m)])
        else:
            resolved_dirs.append(item)

    if not resolved_dirs and all(isinstance(x, dict) for x in dir_list):
        # List of stream kwargs dicts
        return [Stream(**s) for s in dir_list]

    # Validate directory paths and discover MDS roots containing index.json
    valid_mds_dirs = []
    for d in resolved_dirs:
        if os.path.isdir(d):
            index_path = os.path.join(d, 'index.json')
            if os.path.exists(index_path):
                valid_mds_dirs.append(d)
            else:
                # Search subdirectories 1 level deep (e.g. /kaggle/input/dataset/train)
                sub_indices = glob.glob(os.path.join(d, '*', 'index.json'))
                if sub_indices:
                    for si in sub_indices:
                        valid_mds_dirs.append(os.path.dirname(si))
                else:
                    # Still add directory, MDS will validate upon init
                    valid_mds_dirs.append(d)
        else:
            print(f"[Warning] MDS source path does not exist or is not a directory: {d}")

    if not valid_mds_dirs:
        raise ValueError(f"No valid MDS dataset directories discovered from sources: {sources}")

    num_streams = len(valid_mds_dirs)
    if default_proportions is not None and len(default_proportions) == num_streams:
        proportions = default_proportions
    else:
        # Equal proportional weighting across all discovered streams
        proportions = [1.0 / num_streams] * num_streams

    streams = []
    for directory, prop in zip(valid_mds_dirs, proportions):
        streams.append(Stream(local=directory, proportion=prop))

    return streams


class MDSLargeScaleDepthDataset(StreamingDataset):
    """
    MosaicML Streaming dataset for ZipDepth.

    Reads chunked .mds shards in-place from local or mounted directories.
    Applies ZipDepth augmentations (RandomCrop / Resize OneOf) and yields
    standardized dictionary batches compatible with ZipDepthTrainer.
    """

    def __init__(
        self,
        streams=None,
        local=None,
        transform=None,
        batch_size=16,
        shuffle=True,
        cache_limit=None,
        keep_zip=False,
        **kwargs
    ):
        if not STREAMING_AVAILABLE:
            raise ImportError(
                "mosaicml-streaming is required for MDSLargeScaleDepthDataset. "
                "Install via: pip install mosaicml-streaming"
            )

        if streams is None and local is None:
            raise ValueError("Either 'streams' or 'local' must be provided.")

        self.transform = transform
        self._batch_size = batch_size

        super().__init__(
            streams=streams,
            local=local,
            shuffle=shuffle,
            batch_size=batch_size,
            cache_limit=cache_limit,
            keep_zip=keep_zip,
            **kwargs
        )

        # In hybrid setups (local mounted + cloud streaming), subtract read-only local mounts
        # (/kaggle/input) from cache_usage so cache_limit strictly bounds the remote downloads.
        if getattr(self, 'cache_limit', None) and getattr(getattr(self, '_unique_rank_world', None), 'is_local_leader', True):
            readonly_bytes = 0
            for shard_id, shard in enumerate(getattr(self, 'shards', [])):
                if self._is_readonly_shard(shard_id):
                    size_fn = getattr(shard, 'get_raw_size', None)
                    readonly_bytes += size_fn() if callable(size_fn) else getattr(shard, 'raw_size', 0)
            for stream in getattr(self, 'streams', []):
                if getattr(stream, 'remote', None) is None or '/kaggle/input' in str(getattr(stream, 'local', '')):
                    index_fn = getattr(stream, 'get_index_size', None)
                    if callable(index_fn):
                        readonly_bytes += index_fn()
            if readonly_bytes > 0:
                self.cache_usage = max(0, self.cache_usage - readonly_bytes)

    def _is_readonly_shard(self, shard_id):
        """Returns True if the shard is on a read-only filesystem or local mount (not downloaded)."""
        if hasattr(self, 'stream_per_shard') and hasattr(self, 'streams'):
            try:
                stream_id = self.stream_per_shard[shard_id]
                stream = self.streams[stream_id]
                if getattr(stream, 'remote', None) is None or '/kaggle/input' in str(getattr(stream, 'local', '')):
                    return True
            except Exception:
                pass

        if hasattr(self, 'shards') and shard_id < len(self.shards):
            shard = self.shards[shard_id]
            dirname = str(getattr(shard, 'dirname', ''))
            raw_fn = str(getattr(shard, 'raw_filename', ''))
            if '/kaggle/input' in dirname or '/kaggle/input' in raw_fn:
                return True

        return False

    def _evict_coldest_shard(self):
        """Evicts the coldest shard ONLY from downloadable remote streams.
        Local read-only mounts (e.g. /kaggle/input) are strictly protected."""
        if _ShardState is None:
            return

        states = self._shard_states.numpy()
        access_times = self._shard_access_times.numpy()

        local_shard_ids = np.where(states == _ShardState.LOCAL)[0]
        evictable_shard_ids = [
            i for i in local_shard_ids
            if not self._is_readonly_shard(i)
        ]

        if not evictable_shard_ids:
            return

        evictable_shard_ids = np.array(evictable_shard_ids, dtype=np.int64)
        local_shard_times = access_times[evictable_shard_ids]
        coldest_shard_id = int(evictable_shard_ids[np.argmin(local_shard_times)])
        self._evict_shard(coldest_shard_id)

    def _evict_shard(self, shard_id):
        """Safely evict a remote shard while preventing deletion of read-only local mounts."""
        if self._is_readonly_shard(shard_id):
            return

        super()._evict_shard(shard_id)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        # 1. Retrieve domain and key
        domain = sample.get('domain', 'unknown')
        key = sample.get('key', f'sample_{idx}')

        # 2. Decode RGB
        rgb_raw = sample.get('rgb')
        if rgb_raw is None:
            raise KeyError(f"Sample at index {idx} does not contain 'rgb' field.")
        rgb = _decode_rgb(rgb_raw)

        # 3. Decode Depth (if present)
        depth = None
        depth_raw = sample.get('depth')
        if depth_raw is not None:
            depth = _decode_depth(depth_raw)

        # 4. Apply transforms
        if self.transform is not None:
            rgb, depth = self.transform(rgb, depth)

        # 5. Format tensors matching LargeScaleDepthDataset
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

        output = {
            'image': rgb_tensor,
            'domain': str(domain),
            'path': str(key),
        }

        if depth is not None:
            depth_tensor = torch.from_numpy(depth).unsqueeze(0).contiguous()
            output['depth'] = depth_tensor

        return output


def create_mds_dataloader(
    sources,
    height=512,
    width=512,
    batch_size=16,
    num_workers=4,
    shuffle=True,
    cache_limit=None,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
    prefetch_factor=2,
    transform=None,
    **kwargs
):
    """
    Factory function to create a fully configured PyTorch DataLoader
    backed by MosaicML Streaming shards.

    Handles multi-stream composition, DDP rank sharding (built into StreamingDataset),
    and ZipDepth transform wrapping.
    """
    if isinstance(sources, list) and all(isinstance(s, Stream) for s in sources):
        streams = sources
    else:
        streams = build_streams_from_sources(sources)

    if transform is None:
        transform = get_train_transforms(height=height, width=width)

    # If remote cloud streams are present, ensure cache_limit is set (default '15gb')
    # to protect Kaggle disk space from overflowing (COCO ~113GB + BDD100K ~43GB)
    has_remote = any(getattr(s, 'remote', None) is not None for s in streams)
    if has_remote and cache_limit is None:
        cache_limit = "15gb"
        print("[Info] Remote cloud streams detected without cache_limit. Setting default cache_limit='15gb' to protect local disk.")

    if has_remote and "predownload" not in kwargs:
        kwargs["predownload"] = max(batch_size, 32)

    dataset = MDSLargeScaleDepthDataset(
        streams=streams,
        transform=transform,
        batch_size=batch_size,
        shuffle=shuffle,
        cache_limit=cache_limit,
        **kwargs
    )

    # Note: StreamingDataset natively coordinates sample splitting across DDP ranks.
    # Therefore, we do NOT use an external DistributedSampler.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Shuffling is handled internally by StreamingDataset
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0 and persistent_workers),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    return loader, dataset
