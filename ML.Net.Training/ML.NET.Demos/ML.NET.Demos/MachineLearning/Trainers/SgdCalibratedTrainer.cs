using BinaryClassification.MachineLearning.Common;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;

namespace BinaryClassification.MachineLearning.Trainers
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
