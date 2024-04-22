from scipy import spatial
import pandas as pd
import numpy as np
from skimage import measure 
from stardist.matching import matching_dataset
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import csv
from tifffile import imread
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import  normalized_root_mse as mse
import seaborn as sns
from natsort import natsorted
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
        
        
      


"""
predictions: csv file of predictions as a list for different models

groundtruth: csv file of ground truth as a list of TZYX co ordinates (approx/exact centroids)


thresholdscore: veto for score to count true, false positives and false negatives

thresholdspace: tolerance for veto in space

thresholdtime: tolerance for veto in time
"""
class ClassificationScore:
    
    def __init__(self, 
                 predictions: str, 
                 groundtruth: str, 
                 thresholdscore: float = 1 -  1.0E-8,  
                 thresholdspace: int = 20, 
                 thresholdtime: int = 2, 
                 metric: str = 'Euclid',
                 ignorez: bool = False):

         #A list of all the prediction csv files, path object
         if isinstance(predictions, str):
             self.predictions = [Path(predictions)]
         else:
             self.predictions = list(Path(predictions).glob('*.csv')) 
           
         #Approximate locations of the ground truth, Z co ordinate wil be ignored
         self.groundtruth = groundtruth
         self.thresholdscore = thresholdscore
         self.thresholdspace = thresholdspace 
         self.thresholdtime = thresholdtime
         self.ignorez = ignorez
         self.metric = metric
         self.location_pred = []
         self.location_gt = []

         self.dicttree = {}

 
             

    def model_scorer(self):

         
         Name = []
         TP = []
         FP = []
         FN = []
         GT = []
         Precision = []
         Recall = []
         F1 = []
         Pred = []
         columns = ['Model Name', 'True Positive', 'False Positive', 'False Negative', 'Total Predictions', 'GT predictions', 'Precision', 'Recall', 'F1']
         

         dataset_gt  = pd.read_csv(self.groundtruth, delimiter = ',')
         self.location_gt = dataset_gt.iloc[:, :4].astype(int).values.tolist() 
                 
        
         for csv_pred in self.predictions:
            self.location_pred = []
            self.listtime_pred = []
            self.listy_pred = []
            self.listx_pred = []
            self.listscore_pred = []
            self.csv_pred = csv_pred
            name = self.csv_pred.stem
            dataset_pred  = pd.read_csv(self.csv_pred, delimiter = ',')
            
            for index, row in dataset_pred.iterrows():
                T_pred = int(row.iloc[0])
                current_point = (row.iloc[1], row.iloc[2], row.iloc[3])
                score = row.iloc[4] if len(row) > 4 else 1
                
                if score >= float(self.thresholdscore): 
                    self.location_pred.append([int(T_pred), int(row.iloc[1]), int(row.iloc[2]), int(row.iloc[3])])
                    self.listtime_pred.append(int(T_pred))   
                       
            tp, fn, fp, pred, gt = self._TruePositives()
            
            Name.append(name)
            TP.append(tp)
            FN.append(fn)
            FP.append(fp)
            GT.append(gt)
            Pred.append(pred)
            precision = tp/(tp + fp) if (tp + fp) > 0 else 0
            recall = tp/(tp + fn) if (tp + fn) > 0 else 0
            Precision.append(precision)
            Recall.append(recall)
            F1.append(2 * (precision * recall)/(precision + recall))

         data = list(zip(Name, TP, FP, FN, Pred, GT, Precision, Recall, F1))
         data = sorted(data, key = lambda x: x[-2])
         df = pd.DataFrame(data, columns=columns)
         df.to_csv(str(self.csv_pred.parent) + '_model_accuracy' + '.csv')
         return df

     

    def _TruePositives(self):

            tp = 0
            fp = 0
            tree = spatial.cKDTree(self.location_gt)
            for i in range(len(self.location_pred)):
                
                return_index = self.location_pred[i]
                closestpoint = tree.query(return_index)
                spacedistance, timedistance = _TimedDistance(return_index, self.location_gt[closestpoint[1]], self.metric, self.ignorez)
                    
                if spacedistance < self.thresholdspace and timedistance < self.thresholdtime:
                        tp  = tp + 1
                else:
                        fp = fp + 1        
            tp = tp 
            fp = fp 
            fn = self._FalseNegatives()
            return tp, fn, fp, len(self.location_pred), len(self.location_gt)
        

    def _FalseNegatives(self):
        
                        tree = spatial.cKDTree(self.location_pred)
                        fn = 0
                        for i in range(len(self.location_gt)):
                            
                            return_index = (int(self.location_gt[i][0]),int(self.location_gt[i][1]),
                                            int(self.location_gt[i][2]), int(self.location_gt[i][3]))
                            closestpoint = tree.query(return_index)
                            spacedistance, timedistance = _TimedDistance(return_index, self.location_pred[closestpoint[1]], self.metric, self.ignorez)

                            if spacedistance > self.thresholdspace or timedistance > self.thresholdtime:
                                    fn  = fn + 1
                        fn = fn
                        return fn
                    
                                
def _EuclidMetric(x: float,y: float):
    
    return (x - y) * (x - y) 

def _MannhatanMetric(x: float,y: float):
    
    return np.abs(x - y)

def _EuclidSum(func):
    
    return float(np.sqrt( np.sum(func)))

def _ManhattanSum(func):
    
    return float(np.sum(func))

def _general_dist_func(metric, ignorez: bool):
     
     if ignorez:
         start_dim = 2
     else:
         start_dim = 1    
     return lambda x,y : [metric(x[i], y[i]) for i in range(start_dim,len(x))]
 
def _TimedDistance(pointA: tuple, pointB: tuple, metric, ignorez: bool):
     
     if metric == 'Euclid':
        dist_func = _general_dist_func(_EuclidMetric, ignorez)
        spacedistance = _EuclidSum(dist_func(pointA, pointB))
     if metric == 'Manhattan':
        dist_func = _general_dist_func(_MannhatanMetric, ignorez)
        spacedistance = _ManhattanSum(dist_func(pointA, pointB))    
     else:
        dist_func = _general_dist_func(_EuclidMetric, ignorez)
        spacedistance = _EuclidSum(dist_func(pointA, pointB))   
     
     timedistance = float(np.abs(pointA[0] - pointB[0]))
     
     return spacedistance, timedistance