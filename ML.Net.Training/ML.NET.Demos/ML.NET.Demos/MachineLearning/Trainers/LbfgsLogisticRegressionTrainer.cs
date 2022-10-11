using BinaryClassification.MachineLearning.Common;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;

namespace BinaryClassification.MachineLearning.Trainers
{
    public class LbfgsLogisticRegressionTrainer : TrainerBase<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
    {
        public LbfgsLogisticRegressionTrainer() : base()
        {
            Name = "LBFGS Logistic Regression";
            Model = MlContext
                        .BinaryClassification
                        .Trainers
                        .LbfgsLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
        }
    }
}
