using SentimentAnalysis.MachineLearning.Common;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;

namespace SentimentAnalysis.MachineLearning.Trainers
{
    public class SdcaLogicsticRegressionTrainer : TrainerBase<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
    {
        public SdcaLogicsticRegressionTrainer() : base()
        {
            Name = "Sdca Logistic Regression";
            Model = MlContext
                        .BinaryClassification
                        .Trainers
                        .SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
        }
    }
}
