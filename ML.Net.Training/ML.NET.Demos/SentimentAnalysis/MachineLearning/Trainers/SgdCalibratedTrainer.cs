using SentimentAnalysis.MachineLearning.Common;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;

namespace SentimentAnalysis.MachineLearning.Trainers
{
    public class SgdCalibratedTrainer : TrainerBase<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
    {
        public SgdCalibratedTrainer() : base()
        {
            Name = "Sgd Calibrated";
            Model = MlContext
                        .BinaryClassification
                        .Trainers
                        .SgdCalibrated(labelColumnName: "Label", featureColumnName: "Features");
        }
    }
}
