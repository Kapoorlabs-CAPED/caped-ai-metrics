from scipy import spatial
import pandas as pd
import numpy as np
from stardist.matching import matching_dataset
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from tifffile import imread
import seaborn as sns
from natsort import natsorted
from scipy.optimize import linear_sum_assignment
import os


class SegmentationScore:
    """
    ground_truth: Input the directory contianing the ground truth label tif files
    
    predictions: Input the directory containing the predictions tif files (VollSeg/StarDist)
    
    results_dir: Input the name of the directory to store the results in
    
    pattern: In case the input images are not tif files, input the format here
    
    taus: The list of thresholds for computing the metrics 
    
    """
    def __init__(self, ground_truth_dir, predictions_dir, results_dir, acceptable_formats = [".tif", ".TIFF", ".TIF", ".png"], taus=[ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):

        self.ground_truth = []
        self.predictions = []

        sorted_ground_truth = natsorted(os.listdir(ground_truth_dir))
        for fname in sorted_ground_truth:
            if any(fname.endswith(f) for f in acceptable_formats):
                self.ground_truth.append(imread(os.path.join(ground_truth_dir, fname), dtype=np.uint16))
                self.predictions.append(imread(os.path.join(predictions_dir, fname), dtype=np.uint16))
     
        assert len(self.ground_truth) == len(self.predictions), "Number of ground truth and prediction files do not match"
        print(f"Number of images: {len(self.ground_truth)}")
        self.results_dir = results_dir
        self.taus = taus
            
    def seg_stats(self):




        all_stats = []

        # Iterate over each image pair
        for i in tqdm(range(len(self.ground_truth))):
            assert self.ground_truth[i].shape == self.predictions[i].shape, "Images could not be resized properly"
            
            # Calculate statistics for the current image pair
            stats = [matching_dataset(self.ground_truth[i], self.predictions[i], thresh=t, show_progress=False) for t in tqdm(self.taus)]
            
            # Append the statistics to the list
            all_stats.append(stats)

        # Create a figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(15, 15))

        # Plot metrics
        for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'panoptic_quality'):
            mean_stats = []
            for stats in all_stats:
               
                mean_stats.append( [s._asdict()[m] for s in stats])
            
            mean_stats = np.mean(mean_stats, axis=0)
            axes[0].plot(self.taus, mean_stats, '.-', lw=2, label=m)

        axes[0].set_xlabel(r'IoU threshold $\tau$')
        axes[0].set_ylabel('Metric value')
        axes[0].grid()
        axes[0].legend()

        # Plot number statistics
        for m in ('fp', 'tp', 'fn'):
            mean_stats = []
            for stats in all_stats:
                mean_stats.append( [s._asdict()[m] for s in stats])

            mean_stats = np.mean(mean_stats, axis=0) 
            axes[1].plot(self.taus, mean_stats, '.-', lw=2, label=m)

        axes[1].set_xlabel(r'IoU threshold $\tau$')
        axes[1].set_ylabel('Number #')
        axes[1].grid()
        axes[1].legend()

        # Save and show the plot
        plt.tight_layout()
        plt.savefig(self.results_dir + 'metrics_combined.png', dpi=300)
        
        plt.show()       
        
class ClassificationScore:
    """
    pred_csv:     Path to one CSV of predicted centroids (T, Z, Y, X[, score])
    gt_csv:       Path to ground truth CSV (T, Z, Y, X)
    score_thresh: Minimum score to include a prediction
    space_thresh: Max spatial distance (units) for a match
    time_thresh:  Max temporal distance (frames) for a match
    metric:       'Euclid' or 'Manhattan'
    ignore_z:     If True, ignore the Z coordinate
    """
    def __init__(self,
                 pred_csv: str,
                 gt_csv: str,
                 score_thresh: float = 1.0 - 1.0e-6,
                 space_thresh: float = 20,
                 time_thresh: int = 2,
                 metric: str = 'Euclid',
                 ignore_z: bool = False):
        self.pred_path = Path(pred_csv)
        self.pred_df = pd.read_csv(pred_csv)
        self.gt_pts  = pd.read_csv(gt_csv).iloc[:, :4].values.astype(int)
        self.score_thresh = score_thresh
        self.space_thresh = space_thresh
        self.time_thresh  = time_thresh
        self.metric       = metric
        self.ignore_z     = ignore_z

    def _timed_distance(self, a, b):
        # a, b are [T, Z, Y, X]
        if self.ignore_z:
            diffs = a[2:] - b[2:]
        else:
            diffs = a[1:] - b[1:]
        if self.metric == 'Manhattan':
            sd = np.abs(diffs).sum()
        else:
            sd = np.linalg.norm(diffs)
        td = abs(int(a[0]) - int(b[0]))
        return sd, td

    def model_scorer(self):
        # filter by score if present
        df = self.pred_df.copy()
        if df.shape[1] > 4:
            df = df[df.iloc[:,4] >= self.score_thresh]
        pred_pts = df.iloc[:, :4].values.astype(int)

        # Build KD-trees
        gt_tree   = spatial.cKDTree(self.gt_pts)
        pred_tree = spatial.cKDTree(pred_pts) if len(pred_pts)>0 else None

        # Count TP / FP
        tp = fp = 0
        for p in pred_pts:
            dist, idx = gt_tree.query(p)
            sd, td = self._timed_distance(p, self.gt_pts[idx])
            if sd <= self.space_thresh and td <= self.time_thresh:
                tp += 1
            else:
                fp += 1

        # Count FN
        fn = 0
        if pred_tree is not None:
            for g in self.gt_pts:
                dist, idx = pred_tree.query(g)
                sd, td = self._timed_distance(g, pred_pts[idx])
                if sd > self.space_thresh or td > self.time_thresh:
                    fn += 1
        else:
            fn = len(self.gt_pts)

        # Metrics
        precision = tp / (tp + fp) if (tp + fp)>0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn)>0 else 0.0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0

        # Output
        result = pd.DataFrame([{
            'Model': self.pred_path.stem,
            'TP': tp, 'FP': fp, 'FN': fn,
            'Precision': precision, 'Recall': recall, 'F1': f1
        }])
        out_csv = self.pred_path.with_name(f"{self.pred_path.stem}_accuracy.csv")
        result.to_csv(out_csv, index=False)
        return result