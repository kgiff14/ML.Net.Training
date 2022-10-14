using RandomForestRegression.MachineLearning.Common;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers.FastTree;

namespace RandomForestRegression.MachineLearning.Trainers
{
    public class RandomForestTrainer : TrainerBase<FastForestRegressionModelParameters>
    {
        public RandomForestTrainer(int numberOfLeaves, int numberOfTrees) : base()
        {
            Name = $"Random Forest-{numberOfLeaves}-{numberOfTrees}";
            Model = MlContext
                        .Regression
                        .Trainers
                        .FastForest(numberOfLeaves: numberOfLeaves, numberOfTrees: numberOfTrees);
        }
    }
}
