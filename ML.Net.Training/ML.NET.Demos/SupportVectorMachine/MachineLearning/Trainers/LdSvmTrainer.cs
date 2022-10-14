using Microsoft.ML.Trainers;
using SupportVectorMachineClassification.MachineLearning.Common;

namespace SupportVectorMachineClassification.MachineLearning.Trainers
{
    public class LdSvmTrainer : TrainerBase<LdSvmModelParameters>
    {
        public LdSvmTrainer(int treeDepth) : base()
        {
            Name = $"Ld Svm-{treeDepth}";
            Model = MlContext
                        .BinaryClassification
                        .Trainers
                        .LdSvm(treeDepth: treeDepth);
        }
    }
}
