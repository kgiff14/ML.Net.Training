using Microsoft.ML.Trainers;
using Clustering.MachineLearning.Common;

namespace Clustering.MachineLearning.Trainers
{
    public class KMeansTrainer : TrainerBase<KMeansModelParameters>
    {
        public KMeansTrainer(int numberOfClusters) : base()
        {
            Name = $"K Means - {numberOfClusters}";
            Model = MlContext
                        .Clustering
                        .Trainers
                        .KMeans(numberOfClusters: numberOfClusters, featureColumnName: "Features");
        }
    }
}
