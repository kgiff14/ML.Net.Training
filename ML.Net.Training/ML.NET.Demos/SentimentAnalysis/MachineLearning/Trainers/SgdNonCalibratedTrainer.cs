using SentimentAnalysis.MachineLearning.Common;
using Microsoft.ML.Trainers;

namespace SentimentAnalysis.MachineLearning.Trainers
{
    public class SgdNonCalibratedTrainer : TrainerBase<LinearBinaryModelParameters>
    {
        public SgdNonCalibratedTrainer() : base()
        {
            Name = "Sgd NonCalibrated";
            Model = MlContext
                        .BinaryClassification
                        .Trainers
                        .SdcaNonCalibrated(labelColumnName: "Label", featureColumnName: "Features");
        }
    }
}
