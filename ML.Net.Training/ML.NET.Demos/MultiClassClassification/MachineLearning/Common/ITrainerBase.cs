namespace MultiClassClassification.MachineLearning.Common
{
    public interface ITrainerBase
    {

        string Name { get; }

        void Fit(string fileName);

        MulticlassClassificationMetrics Evalute();

        void Save();
    }
}
