using BinaryClassification.MachineLearning.Common;
using Microsoft.ML.Trainers;

namespace BinaryClassification.MachineLearning.Trainers
{
    public class PriorTrainer : TrainerBase<PriorModelParameters>
    {
        public PriorTrainer() : base()
        {
            Name = "Prior";
            Model = MlContext
                        .BinaryClassification
                        .Trainers
                        .Prior(labelColumnName: "Label");
        }
    }
}
