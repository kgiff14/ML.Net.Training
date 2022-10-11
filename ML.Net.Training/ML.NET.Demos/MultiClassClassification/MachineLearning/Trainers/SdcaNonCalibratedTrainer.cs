using MultiClassClassification.MachineLearning.Common;
using Microsoft.ML.Trainers;

namespace MultiClassClassification.MachineLearning.Trainers
{
    public class SdcaNonCalibratedTrainer : TrainerBase<LinearMulticlassModelParameters>
    {
        public SdcaNonCalibratedTrainer() : base()
        {
            Name = "Sdca NonCalibrated";
            Model = MlContext
                        .MulticlassClassification
                        .Trainers
                        .SdcaNonCalibrated(labelColumnName: "Label", featureColumnName: "Features");
        }
    }
}
