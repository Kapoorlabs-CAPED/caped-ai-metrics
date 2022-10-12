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
class SegmentationScore:
    """
    ground_truth: Input the directory contianing the ground truth label tif files
    
    predictions: Input the directory containing the predictions tif files (VollSeg/StarDist)
    
    results_dir: Input the name of the directory to store the results in
    
    pattern: In case the input images are not tif files, input the format here
    
    taus: The list of thresholds for computing the metrics 
    
    """
    def __init__(self, ground_truth, predictions, results_dir, pattern = '*.tif', taus = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  ):
        
        
        self.ground_truth = list(map( imread, list(Path(ground_truth).glob(pattern))))
        self.predictions = list(map( imread, list(Path(predictions).glob(pattern))))
        self.results_dir = results_dir
        self.taus = taus 
        
        
    def seg_stats(self):
        
        stats = [matching_dataset(self.ground_truth, self.predictions, thresh = t, show_progress = False) for t in tqdm(taus)]    
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(25,10))

        for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'panoptic_quality'):
            ax1.plot(self.taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        ax1.set_xlabel(r'IoU threshold $\tau$')
        ax1.set_ylabel('Metric value')
        ax1.grid()

        for m in ('fp', 'tp', 'fn'):
            ax2.plot(self.taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
        ax2.set_xlabel(r'IoU threshold $\tau$')
        ax2.set_ylabel('Number #')
        ax2.grid()
        ax2.legend()
        plt.savefig(self.results_dir + 'AugSeg', dpi=300)
        
        stats_mse = []
        stats_mse_name = self.results_dir + '/' + 'mean_squared_error'
        mse_csv_writer = csv.writer(open(stats_mse_name + '.csv', 'a'))
        mse_csv_writer.writerow(['mean_squared_error'])
        for i in range(len(self.predictions)):
            mse_score = mse(self.ground_truth[i], (self.predictions[i] > 0 ) )
            stats_mse.append(mse_score)
            mse_csv_writer.writerow([mse_score]) 
            
        stats_ssim = []   
        stats_ssim_name = self.results_dir + '/' + 'structural_similarity_index'
        ssim_csv_writer = csv.writer(open(stats_ssim_name + '.csv', 'a'))
        ssim_csv_writer.writerow(['structural_similarity_index'])
        for i in range(len(self.predictions)):
            ssim_score = ssim(self.ground_truth[i], (self.predictions[i] > 0 ) )
            stats_ssim.append(ssim_score)
            ssim_csv_writer.writerow([ssim_score]) 
        
        df = pd.DataFrame(list(zip(stats_mse )), index = None,
                                                    columns =["mean_squared_error"])
        sns.set(style="whitegrid")
        g = sns.violinplot(data=df, orient ='v')
        fig = g.get_figure()
        fig.savefig(self.results_dir  + "mean_squared_error.png", dpi=300)
        
        df = pd.DataFrame(list(zip(stats_ssim )), index = None,
                                                    columns =["structural_similarity_index"])
        sns.set(style="whitegrid")
        g = sns.violinplot(data=df, orient ='v')
        fig = g.get_figure()
        fig.savefig(self.results_dir  + "structural_similarity_index.png", dpi=300)


"""
predictions: csv file of predictions as a list for different models

groundtruth: csv file of ground truth as a list of TZYX co ordinates (approx/exact centroids)

segimage: segmentation image to refine the ground truth locations

thresholdscore: veto for score to count true, false positives and false negatives

thresholdspace: tolerance for veto in space

thresholdtime: tolerance for veto in time
"""
class ClassificationScore:
    
    def __init__(self, predictions, groundtruth, segimage, thresholdscore = 1 -  1.0E-6,  thresholdspace = 10, thresholdtime = 2):

         #A list of all the prediction csv files, path object
         self.predictions = list(Path(predictions).glob('*.csv')) 
         #Segmentation image for accurate metric evaluation
         self.segimage = imread(segimage)
         #Approximate locations of the ground truth, Z co ordinate wil be ignored
         self.groundtruth = groundtruth
         self.thresholdscore = thresholdscore
         self.thresholdspace = thresholdspace 
         self.thresholdtime = thresholdtime
         self.location_pred = []
         self.location_gt = []

         self.dicttree = {}

    def _make_trees(self):
        
        for i in range(self.segimage.shape[0]):
            #Make a ZYX image
            currentimage = self.segimage[i,:,:,:]
            props = measure.regionprops(currentimage)
            indices = [prop.centroid for prop in props]
            if len(indices) > 0:
                tree = spatial.cKDTree(indices)
                self.dicttree[int(i)] = [tree, indices] 
             

    def model_scorer(self):

         self._make_trees()
         Name = []
         TP = []
         FP = []
         FN = []
         GT = []
         Pred = []
         columns = ['Model Name', 'True Positive', 'False Positive', 'False Negative', 'Total Predictions', 'GT predictions']
         

         dataset_gt  = pd.read_csv(self.groundtruth, delimiter = ',')
         for index, row in dataset_gt.iterrows():
              T_gt = int(row[0])
              current_point = (row[1], row[2], row[3])
              tree, indices = self.dicttree[int(T_gt)]
              distance, nearest_location = tree.query(current_point)
              nearest_location = (int(indices[nearest_location][0]), int(indices[nearest_location][1]), int(indices[nearest_location][2]))
              self.location_gt.append([T_gt, nearest_location[0], nearest_location[1], nearest_location[2]])
        
         

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
              T_pred = int(row[0])
              current_point = (row[1], row[2], row[3])
              score = row[4]
              if score >= float(self.thresholdscore): 
                  self.location_pred.append([int(T_pred), int(row[1]), int(row[2]), int(row[3])])
              
            tp, fn, fp, pred, gt = self._TruePositives()
            
            Name.append(name)
            TP.append(tp)
            FN.append(fn)
            FP.append(fp)
            GT.append(gt)
            Pred.append(pred)
         data = list(zip(Name, TP, FP, FN, Pred, GT))

         df = pd.DataFrame(data, columns=columns)
         df.to_csv(str(self.csv_pred.parent) + '_Model_Accuracy' + '.csv')
         return df

     

    def _TruePositives(self):

            tp = 0
            fp = 0
            tree = spatial.cKDTree(self.location_gt)
            for i in range(len(self.location_pred)):
                
                return_index = self.location_pred[i]
                closestpoint = tree.query(return_index)
                spacedistance, timedistance = _TimedDistance(return_index, self.location_gt[closestpoint[1]])
                    
                if spacedistance < self.thresholdspace and timedistance < self.thresholdtime:
                        tp  = tp + 1
                else:
                        fp = fp + 1        
            
            fn = self._FalseNegatives()
            return tp, fn, fp, len(self.location_pred), len(self.location_gt)
        

    def _FalseNegatives(self):
        
                        tree = spatial.cKDTree(self.location_pred)
                        fn = 0
                        for i in range(len(self.location_gt)):
                            
                            return_index = (int(self.location_gt[i][0]),int(self.location_gt[i][1]),
                                            int(self.location_gt[i][2]), int(self.location_gt[i][3]))
                            closestpoint = tree.query(return_index)
                            spacedistance, timedistance = _TimedDistance(return_index, self.location_pred[closestpoint[1]])

                            if spacedistance > self.thresholdspace or timedistance > self.thresholdtime:
                                    fn  = fn + 1

                        return fn
                    
                                
def _EuclidMetric(x,y):
    
    return (x - y) * (x - y) 

def _MannhatanMetric(x,y):
    
    return np.abs(x - y)

def _EuclidSum(func):
    
    return float(np.sqrt( np.sum(func)))

def _ManhattanSum(func):
    
    return float(np.sum(func))

def _general_dist_func(metric):
     
     return lambda x,y : [metric(x[i], y[i]) for i in range(1,len(x))]
 
def _TimedDistance(pointA, pointB):

     dist_func = _general_dist_func(_EuclidMetric)
     
     spacedistance = _EuclidSum(dist_func(pointA, pointB))
     
     timedistance = float(np.abs(pointA[0] - pointB[0]))
     
     return spacedistance, timedistance