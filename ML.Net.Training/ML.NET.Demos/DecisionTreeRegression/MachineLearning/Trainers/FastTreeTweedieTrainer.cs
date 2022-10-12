using DecisionTreeRegression.MachineLearning.Common;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers.FastTree;

namespace DecisionTreeRegression.MachineLearning.Trainers
{
    public class FastTreeTweedieTrainer : TrainerBase<FastTreeTweedieModelParameters>
    {
        public FastTreeTweedieTrainer(int numberOfLeaves, int numberOfTrees, double learningRate = 0.2) : base()
        {
            Name = $"Fast Tree Tweedie-{numberOfLeaves}-{numberOfTrees}-{learningRate}";
            Model = MlContext
                        .Regression
                        .Trainers
                        .FastTreeTweedie(numberOfLeaves: numberOfLeaves, numberOfTrees: numberOfTrees, learningRate: learningRate);
        }
    }
}
