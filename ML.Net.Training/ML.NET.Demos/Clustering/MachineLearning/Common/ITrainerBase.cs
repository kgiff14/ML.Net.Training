namespace Clustering.MachineLearning.Common
{
    public interface ITrainerBase
    {
        string Name { get; }

        void Fit(string fileName);

        ClusteringMetrics Evalute();

        void Save();
    }
}
