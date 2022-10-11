using Microsoft.ML.Trainers;
using MultiClassClassification.MachineLearning.Common;

namespace MultiClassClassification.MachineLearning.Trainers
{
    public class LbfgsMaximumEntropyTrainer : TrainerBase<MaximumEntropyModelParameters>
    {
        public LbfgsMaximumEntropyTrainer() : base()
        {
            Name = "LBFGS Maximum Entropy";
            Model = MlContext
                        .MulticlassClassification
                        .Trainers
                        .LbfgsMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
        }
    }
}
