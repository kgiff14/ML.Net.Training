using DecisionTreeRegression.MachineLearning.Common;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers.FastTree;

namespace DecisionTreeRegression.MachineLearning.Trainers
{
    public class GamTrainer : TrainerBase<GamRegressionModelParameters>
    {
        public GamTrainer() : base()
        {
            Name = "GAM";
            Model = MlContext
                        .Regression
                        .Trainers
                        .Gam();
        }
    }
}
