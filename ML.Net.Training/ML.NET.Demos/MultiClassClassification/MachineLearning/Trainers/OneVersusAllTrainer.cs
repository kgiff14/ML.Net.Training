using Microsoft.ML.Trainers;
using MultiClassClassification.MachineLearning.Common;

namespace MultiClassClassification.MachineLearning.Trainers
{
    public class OneVersusAllTrainer : TrainerBase<OneVersusAllModelParameters>
    {
        public OneVersusAllTrainer() : base()
        {
            Name = "One Versus All";
            Model = MlContext
                        .MulticlassClassification
                        .Trainers
                        .OneVersusAll(labelColumnName: "Label", binaryEstimator: MlContext.BinaryClassification.Trainers.SgdCalibrated());
        }
    }
}
