using Microsoft.ML.Trainers;
using MultiClassClassification.MachineLearning.Common;

namespace MultiClassClassification.MachineLearning.Trainers
{
    public class SdcaMaximumEntropyTrainer : TrainerBase<MaximumEntropyModelParameters>
    {
        public SdcaMaximumEntropyTrainer() : base()
        {
            Name = "Sdca Maximum Entropy";
            Model = MlContext
                        .MulticlassClassification
                        .Trainers
                        .SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
        }
    }
}
