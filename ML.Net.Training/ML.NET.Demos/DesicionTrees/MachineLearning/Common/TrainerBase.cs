using DesicionTrees.MachineLearning.DataModels;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace DesicionTrees.MachineLearning.Common
{
    public class TrainerBase<TParamters> : ITrainerBase
        where TParamters : class
    {
        public string Name { get; protected set; }

        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "DecisionTreesModel.mdl");
        protected readonly MLContext MlContext;
        protected DataOperationsCatalog.TrainTestData DataSplit;
        protected ITrainerEstimator<BinaryPredictionTransformer<TParamters>, TParamters> Model;
        protected ITransformer TrainedModel;

        protected TrainerBase()
        {
            MlContext = new(11);
        }

        public BinaryClassificationMetrics Evalute()
        {
            var testSetTransform = TrainedModel.Transform(DataSplit.TestSet);

            return MlContext.BinaryClassification.Evaluate(testSetTransform);
        }

        public void Fit(string fileName)
        {
            if (!File.Exists(fileName))
            {
                throw new FileNotFoundException(fileName);
            }

            DataSplit = LoadAndPrepareData(fileName);
            var pipeline = BuildDataProcessingPipeline();
            var trainingPipeline = pipeline.Append(Model);

            TrainedModel = trainingPipeline.Fit(DataSplit.TrainSet);
        }

        public void Save()
        {
            MlContext.Model.Save(TrainedModel, DataSplit.TrainSet.Schema, ModelPath);
        }

        private DataOperationsCatalog.TrainTestData LoadAndPrepareData(string trainingFileName)
        {
            var trainingDataView = MlContext.Data.LoadFromTextFile<PenguinData>(trainingFileName, hasHeader: true, separatorChar: ',');

            return MlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
        }

        private EstimatorChain<NormalizingTransformer> BuildDataProcessingPipeline()
        {
            var pipeline = MlContext.Transforms.Concatenate("Features",
                                                            nameof(PenguinData.CulmenDepth),
                                                            nameof(PenguinData.CulmenLength))
                            .Append(MlContext.Transforms.NormalizeMinMax("Features", "Features"))
                            .AppendCacheCheckpoint(MlContext);

            return pipeline;
        }
    }
}
