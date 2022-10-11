using BinaryClassification.MachineLearning.Common;
using Microsoft.ML.Trainers;

namespace BinaryClassification.MachineLearning.Trainers
{
    public class SdcaNonCalibratedTrainer : TrainerBase<LinearBinaryModelParameters>
    {
        public SdcaNonCalibratedTrainer() : base()
        {
            Name = "Sdca NonCalibrated";
            Model = MlContext
                        .BinaryClassification
                        .Trainers
                        .SdcaNonCalibrated(labelColumnName: "Label", featureColumnName: "Features");
        }
    }
}
