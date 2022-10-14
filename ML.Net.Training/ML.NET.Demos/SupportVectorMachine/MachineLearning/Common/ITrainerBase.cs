namespace SupportVectorMachineClassification.MachineLearning.Common
{
    public interface ITrainerBase
    {
        string Name { get; }

        void Fit(string fileName);

        BinaryClassificationMetrics Evalute();

        void Save();
    }
}
