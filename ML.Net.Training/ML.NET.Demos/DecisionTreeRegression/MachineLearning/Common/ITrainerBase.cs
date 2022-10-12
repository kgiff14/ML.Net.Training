namespace DecisionTreeRegression.MachineLearning.Common
{
    public interface ITrainerBase
    {
        string Name { get; }

        void Fit(string fileName);

        RegressionMetrics Evalute();

        void Save();
    }
}
