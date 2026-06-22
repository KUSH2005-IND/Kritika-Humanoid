# # storage/unknown_handler.py
# """
# Unknown Face Handler — saves unrecognised face crops for later enrollment.
# Time-based deduplication to avoid flooding disk with similar faces.
# """

# import cv2
# import time
# from pathlib import Path

# import sys
# import os
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from config import UNKNOWN_FACES_DIR


# class UnknownFaceHandler:
#     """
#     Saves unknown face crops for later enrollment.
#     Deduplication: skip save if a very similar face was saved < MIN_INTERVAL_SEC ago.
#     (Simple time-based; can upgrade to embedding-based clustering.)
#     """

#     MIN_INTERVAL_SEC = 10.0

#     def __init__(self, save_dir=UNKNOWN_FACES_DIR):
#         self.save_dir = Path(save_dir)
#         self.save_dir.mkdir(parents=True, exist_ok=True)
#         self._last_saved: dict[int, float] = {}   # track_id → timestamp
#         self._counter = 0

#     def save(self, face_crop, track_id: int = -1) -> str | None:
#         """
#         Save a face crop if enough time has passed since last save.
#         Returns the file path if saved, None otherwise.
#         """
#         if face_crop is None or face_crop.size == 0:
#             return None

#         now = time.time()
#         if now - self._last_saved.get(track_id, 0.0) < self.MIN_INTERVAL_SEC:
#             return None

#         self._counter += 1
#         ts = time.strftime("%Y%m%d_%H%M%S")
#         filename = f"unknown_{ts}_t{track_id}_{self._counter:04d}.jpg"
#         save_path = self.save_dir / filename

#         try:
#             cv2.imwrite(str(save_path), face_crop)
#             self._last_saved[track_id] = now
#             print(f"[Unknown] Saved unknown face: {filename}")
#             return str(save_path)
#         except Exception as e:
#             print(f"[Unknown] Failed to save face crop: {e}")
#             return None




# storage/unknown_handler.py
"""
UnknownPersonManager — clusters and tracks unrecognised face crops.
 
Replaces the original UnknownFaceHandler (which saved flat images to disk with
no grouping, threshold, or operator workflow).
 
Design overview
───────────────
Clusters:
  Each distinct unknown individual gets one UnknownCluster.  All face crops and
  ArcFace embeddings observed for that person accumulate in the cluster.
 
Two lookup paths (per recognition call):
  fast path       active_clusters[track_id]             O(1)
  embedding path  cosine similarity vs cluster.mean_emb  for person re-entering frame
 
Quality gate:
  A crop is saved only when SCRFD confidence, face size, and save-interval
  thresholds are all met — so 20 instances = 20 genuinely usable crops.
 
Operator flow (U-key):
  When a cluster reaches UNKNOWN_ACCUMULATE_TARGET instances its bounding box
  turns amber.  The operator presses U, picks a cluster number, types a name,
  and the cluster is enrolled via FAISS + saved to database/known/<name>/.
 
Auto-purge (The Catch):
  When a recognition run returns is_known=True for a track that owns an active
  cluster, or matches a detached cluster by embedding, the cluster is deleted —
  it was temporarily misidentified known person data.
 
Disk persistence:
  Clusters with ≥ UNKNOWN_MIN_PERSIST_COUNT crops are written to
  unknown_faces/cluster_<id>/ (crops + meta.json) so they survive restarts.
 
Thread safety:
  _lock guards active_clusters, detached_clusters, and all UnknownCluster
  mutations.  The U-key background thread holds _lock only for brief dict/list
  lookups; heavy work (FAISS write, file copies) is done outside _lock.
"""
 
import time
import uuid
import json
import shutil
import threading
import base64
import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
 
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    UNKNOWN_FACES_DIR,
    UNKNOWN_ACCUMULATE_TARGET,
    UNKNOWN_MIN_INTERVAL_SEC,
    UNKNOWN_MIN_FACE_SIZE,
    UNKNOWN_SCRFD_CONF_GATE,
    UNKNOWN_MATCH_THRESHOLD,
    UNKNOWN_PURGE_THRESHOLD,
    UNKNOWN_MIN_PERSIST_COUNT,
)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────
 
@dataclass
class UnknownCluster:
    """
    One cluster = one unknown individual.
 
    Fields that survive a restart (persisted in meta.json):
      cluster_id, display_label, count, first_seen, mean_emb
 
    Fields that are rebuilt at runtime:
      track_ids   — set of ByteTrack IDs that have been mapped here
      crops       — in-memory list of 112×112 BGR ndarrays (also on disk)
      embeddings  — in-memory 512-D ArcFace vectors; empty for disk-loaded clusters
                    (re-embedded from crops at enrollment time)
 
    cancelled:
      Set True by any purge call.  The U-key enrollment thread checks this flag
      before committing to FAISS to handle the race where a cluster is purged
      while the operator is mid-prompt.
    """
    cluster_id:    str
    display_label: str
    track_ids:     set            = field(default_factory=set)
    crops:         list           = field(default_factory=list)
    embeddings:    list           = field(default_factory=list)
    mean_emb:      np.ndarray     = None
    count:         int            = 0
    first_seen:    float          = 0.0
    last_saved:    float          = 0.0
    disk_path:     Path           = None
    cancelled:     bool           = False
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Manager
# ─────────────────────────────────────────────────────────────────────────────
 
class UnknownPersonManager:
    """
    Central manager for all unknown-person clusters.
 
    Primary interface (called from app.py main loop):
      handle_unknown(face_crop, embedding, track_id, scrfd_score, face_bbox)
          → display label string
 
      purge_by_track(track_id, known_name)   — Case 1 catch
      purge_by_embedding(query_emb, known_name)  — Case 2 catch
 
      detach_track(track_id)   — call when ByteTrack times out a track
 
    Operator interface (called from U-key background thread):
      ready_clusters  → list of clusters at UNKNOWN_ACCUMULATE_TARGET
      all_clusters    → all clusters for display
      enroll_cluster(cluster_id, name, embedder, db, known_db_root) → bool
    """
 
    def __init__(self, save_dir: str = UNKNOWN_FACES_DIR):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
 
        self._lock = threading.Lock()
        # Primary lookup: track_id → cluster (O(1) per frame)
        self.active_clusters:   dict[int, UnknownCluster] = {}
        # Fallback: clusters whose tracks have timed out
        self.detached_clusters: list[UnknownCluster]      = []
        # Monotonically increasing label seed (A, B, ..., Z, AA, ...)
        self._label_counter: int = 0
 
        self._load_from_disk()
 
    # ─── Label generator ──────────────────────────────────────────────────────
 
    def _next_label(self) -> str:
        """
        Returns 'Unknown #A', 'Unknown #B', ..., 'Unknown #Z', 'Unknown #AA', ...
        Must be called while holding self._lock.
        """
        n = self._label_counter + 1
        self._label_counter += 1
        letters = ''
        while n > 0:
            n, rem = divmod(n - 1, 26)
            letters = chr(65 + rem) + letters
        return f"Unknown #{letters}"
 
    @staticmethod
    def _label_to_index(label: str) -> int:
        """Inverse of _next_label — converts 'Unknown #AB' back to integer index."""
        if '#' not in label:
            return 0
        letters = label.split('#')[-1].strip()
        n = 0
        for ch in letters:
            n = n * 26 + (ord(ch) - 64)
        return n
 
    # ─── Internal helpers ─────────────────────────────────────────────────────
 
    def _find_detached(self, embedding: np.ndarray) -> 'UnknownCluster | None':
        """
        Best-match search over detached clusters.
        Returns the cluster with highest cosine similarity ≥ UNKNOWN_MATCH_THRESHOLD,
        or None if no match.
        Must be called while holding self._lock.
        """
        best_cluster = None
        best_score   = UNKNOWN_MATCH_THRESHOLD   # minimum to qualify
 
        for cluster in self.detached_clusters:
            if cluster.mean_emb is None:
                continue
            score = float(np.dot(embedding, cluster.mean_emb))
            if score >= best_score:
                best_score   = score
                best_cluster = cluster
 
        return best_cluster
 
    @staticmethod
    def _update_mean(current: 'np.ndarray | None', new_emb: np.ndarray, n: int) -> np.ndarray:
        """
        Incremental normalised mean: mean_{n+1} = (mean_n * n + new) / (n+1), then L2-norm.
        Must be called while holding self._lock.
        """
        if current is None or n == 0:
            return new_emb.astype(np.float32).copy()
        updated = (current * n + new_emb) / (n + 1)
        norm = np.linalg.norm(updated)
        if norm < 1e-10:
            return updated.astype(np.float32)
        return (updated / norm).astype(np.float32)
 
    @staticmethod
    def _passes_quality_gate(
        face_crop:   np.ndarray,
        scrfd_score: float,
        face_bbox:   list,
    ) -> bool:
        """True when the crop is worth keeping: confident detection + large enough face."""
        if face_crop is None or face_crop.size == 0:
            return False
        if scrfd_score < UNKNOWN_SCRFD_CONF_GATE:
            return False
        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            if min(x2 - x1, y2 - y1) < UNKNOWN_MIN_FACE_SIZE:
                return False
        return True
 
    # ─── Crop accumulation ────────────────────────────────────────────────────
 
    def _add_crop(
        self,
        cluster:   UnknownCluster,
        crop:      np.ndarray,
        embedding: np.ndarray,
        now:       float,
    ):
        """
        Append crop and embedding to cluster, update incremental mean, write to disk.
        Must be called while holding self._lock.
        """
        cluster.crops.append(crop.copy())
        cluster.embeddings.append(embedding.copy())
        cluster.mean_emb  = self._update_mean(cluster.mean_emb, embedding, cluster.count)
        cluster.count    += 1
        cluster.last_saved = now
 
        # Persist to disk once we have enough to be worth keeping across restarts.
        # 112×112 JPEG ≈ 3–8 KB; imwrite at this size takes < 1 ms.
        if cluster.count >= UNKNOWN_MIN_PERSIST_COUNT:
            self._write_crop(cluster, crop)
            self._write_meta(cluster)
 
    # ─── Main entry point ─────────────────────────────────────────────────────
 
    def handle_unknown(
        self,
        face_crop:   np.ndarray,
        embedding:   np.ndarray,
        track_id:    int,
        scrfd_score: float,
        face_bbox:   list,
    ) -> str:
        """
        Called from app.py on every Unknown recognition event.
 
        Routing:
          1. Fast path   — track already has an active cluster
          2. Embedding path — match against detached clusters (person re-entered)
          3. New cluster   — create fresh cluster (only if quality gate passes)
 
        Returns:
          The display label for this track (e.g. 'Unknown #B (12)' or 'Unknown').
          The caller should pass this to tracker.update_track_identity() so the
          overlay box shows the correct label.
        """
        quality_ok = self._passes_quality_gate(face_crop, scrfd_score, face_bbox)
        now = time.time()
 
        with self._lock:
            # ── 1. Fast path ──────────────────────────────────────────────────
            if track_id in self.active_clusters:
                cluster = self.active_clusters[track_id]
                if quality_ok and now - cluster.last_saved >= UNKNOWN_MIN_INTERVAL_SEC:
                    self._add_crop(cluster, face_crop, embedding, now)
                return cluster.display_label
 
            # ── 2. Embedding path ─────────────────────────────────────────────
            # Embedding is always available (ArcFace ran already), so we can
            # search detached clusters regardless of the quality gate.
            matched = self._find_detached(embedding)
            if matched:
                matched.track_ids.add(track_id)
                self.active_clusters[track_id] = matched
                self.detached_clusters.remove(matched)
                if quality_ok and now - matched.last_saved >= UNKNOWN_MIN_INTERVAL_SEC:
                    self._add_crop(matched, face_crop, embedding, now)
                return matched.display_label
 
            # ── 3. New cluster ────────────────────────────────────────────────
            # Only create a cluster if this first crop is worth keeping;
            # otherwise show 'Unknown' and wait for a better frame.
            if not quality_ok:
                return 'Unknown'
 
            cid   = uuid.uuid4().hex[:4]
            label = self._next_label()
            dpath = self.save_dir / f"cluster_{cid}"
            dpath.mkdir(parents=True, exist_ok=True)
 
            cluster = UnknownCluster(
                cluster_id    = cid,
                display_label = label,
                track_ids     = {track_id},
                first_seen    = now,
                last_saved    = 0.0,
                disk_path     = dpath,
            )
            self.active_clusters[track_id] = cluster
            self._add_crop(cluster, face_crop, embedding, now)
            return label
 
    # ─── Track lifecycle ──────────────────────────────────────────────────────
 
    def detach_track(self, track_id: int):
        """
        Called when ByteTrack times out a track (person has been gone > TRACK_TIMEOUT_SEC).
        Moves the cluster to detached_clusters so it can be re-matched by embedding
        if the person re-enters frame with a new track_id.
        """
        with self._lock:
            cluster = self.active_clusters.pop(track_id, None)
            if cluster is not None and cluster not in self.detached_clusters:
                self.detached_clusters.append(cluster)
 
    def get_active_track_ids(self) -> list[int]:
        """Snapshot of currently active track IDs — safe to iterate outside _lock."""
        with self._lock:
            return list(self.active_clusters.keys())
 
    # ─── Purge (The Catch) ────────────────────────────────────────────────────
 
    def purge_by_track(self, track_id: int, known_name: str) -> bool:
        """
        Case 1: recognition returns is_known=True for a track that has an active cluster.
        The cluster belongs to a temporarily misidentified known person — purge it.
 
        Returns True if a cluster was purged, False if no cluster was found for this track.
        """
        with self._lock:
            cluster = self.active_clusters.pop(track_id, None)
            if cluster is None:
                return False
            cluster.cancelled = True
            # Also remove from detached list if it was somehow there
            if cluster in self.detached_clusters:
                self.detached_clusters.remove(cluster)
 
        self._delete_from_disk(cluster)
        print(f"[Unknown] Purged {cluster.display_label} — identified as '{known_name}'")
        return True
 
    def purge_by_embedding(self, query_emb: np.ndarray, known_name: str):
        """
        Case 2: recognition returns is_known=True for a track with no active cluster.
        The person may have been in a detached cluster (they left and re-entered).
        Compares query_emb against all detached cluster mean embeddings and purges
        any cluster with similarity ≥ UNKNOWN_PURGE_THRESHOLD.
 
        The purge threshold (0.37) is 0.05 below the recognition threshold (0.42)
        to account for cluster mean embeddings being averaged over suboptimal frames.
        """
        to_purge = []
        with self._lock:
            for cluster in self.detached_clusters:
                if cluster.mean_emb is None:
                    continue
                if float(np.dot(query_emb, cluster.mean_emb)) >= UNKNOWN_PURGE_THRESHOLD:
                    to_purge.append(cluster)
 
            for cluster in to_purge:
                cluster.cancelled = True
                self.detached_clusters.remove(cluster)
                # Remove from active_clusters too if it was re-attached to a new track_id
                for tid, c in list(self.active_clusters.items()):
                    if c is cluster:
                        del self.active_clusters[tid]
 
        for cluster in to_purge:
            self._delete_from_disk(cluster)
            print(f"[Unknown] Purged {cluster.display_label} — identified as '{known_name}'")
 
    # ─── Operator interface ───────────────────────────────────────────────────
 
    @property
    def ready_clusters(self) -> list[UnknownCluster]:
        """Clusters at or above UNKNOWN_ACCUMULATE_TARGET, sorted by first_seen."""
        with self._lock:
            seen, result = set(), []
            for c in list(self.active_clusters.values()) + self.detached_clusters:
                if c.cluster_id not in seen and c.count >= UNKNOWN_ACCUMULATE_TARGET:
                    result.append(c)
                    seen.add(c.cluster_id)
            return sorted(result, key=lambda c: c.first_seen)
 
    @property
    def all_clusters(self) -> list[UnknownCluster]:
        """All clusters (active + detached), deduplicated and sorted by first_seen."""
        with self._lock:
            seen, result = set(), []
            for c in list(self.active_clusters.values()) + self.detached_clusters:
                if c.cluster_id not in seen:
                    result.append(c)
                    seen.add(c.cluster_id)
            return sorted(result, key=lambda c: c.first_seen)
 
    def ready_count(self) -> int:
        """Number of clusters ready for naming. Used by HUD hint."""
        return len(self.ready_clusters)
 
    def get_track_status(self, track_id: int) -> tuple[str, int]:
        """
        Returns (display_label, crop_count) for a track's active cluster.
        Returns ('Unknown', 0) if the track has no cluster assigned.
        Called from draw_overlay() — acquires lock briefly.
        """
        with self._lock:
            c = self.active_clusters.get(track_id)
            return (c.display_label, c.count) if c else ('Unknown', 0)
 
    def enroll_cluster(
        self,
        cluster_id:    str,
        name:          str,
        embedder,
        db,
        known_db_root: str,
    ) -> bool:
        """
        Enroll a ready cluster as a named person.  Called from the U-key background thread.
 
        Steps:
          1. Find cluster, snapshot crops/embeddings under lock.
          2. Build embedding matrix outside lock (can be slow for large clusters).
          3. Check cancelled flag again — handles race with purge.
          4. Add embeddings to FAISS.
          5. Copy crops to database/known/<name>/.
          6. Remove cluster from manager state under lock.
          7. Delete cluster directory from disk.
 
        Returns True on success, False if cluster was purged or not found.
        """
        # Step 1: Find and snapshot
        with self._lock:
            target = next(
                (c for c in list(self.active_clusters.values()) + self.detached_clusters
                 if c.cluster_id == cluster_id),
                None,
            )
            if target is None:
                print(f"[Unknown] Cluster '{cluster_id}' not found — may have been purged.")
                return False
            if target.cancelled:
                print(f"[Unknown] {target.display_label} was identified as a known person "
                      f"— cluster cleared.")
                return False
            crops_snap = list(target.crops)
            embs_snap  = list(target.embeddings)
            label_snap = target.display_label
            cid_snap   = target.cluster_id
 
        # Step 2: Build embedding matrix (outside lock)
        if embs_snap:
            # In-memory embeddings available (cluster was active this session)
            emb_matrix = np.vstack([e.reshape(1, -1) for e in embs_snap])
        elif crops_snap:
            # Disk-loaded cluster: embeddings not persisted, re-embed from crops
            print(f"[Unknown] Re-embedding {len(crops_snap)} crops for {label_snap}...")
            emb_matrix = embedder.embed_batch(crops_snap)
        else:
            print(f"[Unknown] No data for {label_snap}. Aborting enrollment.")
            return False
 
        # Step 3: Final cancellation check
        if target.cancelled:
            print(f"[Unknown] {label_snap} cancelled before FAISS commit.")
            return False
 
        # Step 4: Add to FAISS
        db.add_person(name, emb_matrix)
 
        # Step 5: Copy crops to database/known/<name>/
        person_dir = Path(known_db_root) / name
        person_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        saved = 0
        for i, crop in enumerate(crops_snap):
            out = person_dir / f"unknown_origin_{ts}_{i:03d}.jpg"
            if cv2.imwrite(str(out), crop):
                saved += 1
 
        # Step 6: Remove from manager state
        with self._lock:
            for tid, c in list(self.active_clusters.items()):
                if c.cluster_id == cid_snap:
                    del self.active_clusters[tid]
            if target in self.detached_clusters:
                self.detached_clusters.remove(target)
 
        # Step 7: Delete cluster directory
        self._delete_from_disk(target)
 
        print(f"[Unknown] ✓ '{name}' enrolled from {label_snap} "
              f"({len(embs_snap) if embs_snap else 're-computed'} embeddings, "
              f"{saved}/{len(crops_snap)} crops saved to database/known/{name}/)")
        return True
 
    # ─── Disk I/O ─────────────────────────────────────────────────────────────
 
    def _write_crop(self, cluster: UnknownCluster, crop: np.ndarray):
        """Write the latest crop to disk.  Must be called while holding self._lock."""
        path = cluster.disk_path / f"crop_{cluster.count:03d}.jpg"
        try:
            cv2.imwrite(str(path), crop)
        except Exception as e:
            print(f"[Unknown] Crop write error ({cluster.display_label}): {e}")
 
    def _write_meta(self, cluster: UnknownCluster):
        """Write/overwrite meta.json for the cluster.  Must be called while holding self._lock."""
        meta = {
            'cluster_id':    cluster.cluster_id,
            'display_label': cluster.display_label,
            'count':         cluster.count,
            'first_seen':    cluster.first_seen,
            # mean_emb stored as base64 raw bytes (float32, 512 values = 2048 bytes)
            'mean_emb': (
                base64.b64encode(cluster.mean_emb.astype(np.float32).tobytes()).decode('ascii')
                if cluster.mean_emb is not None else None
            ),
        }
        try:
            with open(cluster.disk_path / 'meta.json', 'w') as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"[Unknown] Meta write error ({cluster.display_label}): {e}")
 
    def _delete_from_disk(self, cluster: UnknownCluster):
        """Remove all files for a cluster.  Safe to call outside _lock."""
        if cluster.disk_path and cluster.disk_path.exists():
            try:
                shutil.rmtree(str(cluster.disk_path))
            except Exception as e:
                print(f"[Unknown] Failed to delete {cluster.disk_path}: {e}")
 
    def _load_from_disk(self):
        """
        Scan unknown_faces/cluster_*/ on startup and load persisted clusters
        into detached_clusters.  Clusters with fewer than UNKNOWN_MIN_PERSIST_COUNT
        crops are too sparse to be useful and are silently discarded.
        """
        for cluster_dir in sorted(self.save_dir.glob('cluster_*')):
            meta_path = cluster_dir / 'meta.json'
            if not meta_path.exists():
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
 
                count = meta.get('count', 0)
                if count < UNKNOWN_MIN_PERSIST_COUNT:
                    continue    # Too sparse — discard
 
                # Reconstruct mean_emb from base64
                mean_emb = None
                if meta.get('mean_emb'):
                    raw      = base64.b64decode(meta['mean_emb'])
                    mean_emb = np.frombuffer(raw, dtype=np.float32).copy()
 
                # Reload crops from disk
                crops = []
                for cf in sorted(cluster_dir.glob('crop_*.jpg')):
                    img = cv2.imread(str(cf))
                    if img is not None:
                        crops.append(img)
 
                cluster = UnknownCluster(
                    cluster_id    = meta['cluster_id'],
                    display_label = meta['display_label'],
                    track_ids     = set(),       # tracks from previous run are gone
                    crops         = crops,
                    embeddings    = [],           # not persisted; re-embedded at enrollment
                    mean_emb      = mean_emb,
                    count         = count,
                    first_seen    = meta.get('first_seen', 0.0),
                    last_saved    = 0.0,
                    disk_path     = cluster_dir,
                )
                self.detached_clusters.append(cluster)
 
                # Advance label counter to avoid issuing duplicate labels this session
                idx = self._label_to_index(meta.get('display_label', ''))
                if idx > self._label_counter:
                    self._label_counter = idx
 
                print(f"[Unknown] Loaded {meta['display_label']} ({count} crops) from disk")
 
            except Exception as e:
                print(f"[Unknown] Failed to load {cluster_dir.name}: {e}")