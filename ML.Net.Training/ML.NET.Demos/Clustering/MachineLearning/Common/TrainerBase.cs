using Clustering.MachineLearning.DataModels;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace Clustering.MachineLearning.Common
{
    public class TrainerBase<TParamters> : ITrainerBase
        where TParamters : class
    {
        public string Name { get; protected set; }

        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "ClusterModel.mdl");
        protected readonly MLContext MlContext;
        protected DataOperationsCatalog.TrainTestData DataSplit;
        protected ITrainerEstimator<ClusteringPredictionTransformer<TParamters>, TParamters> Model;
        protected ITransformer TrainedModel;

        protected TrainerBase()
        {
            MlContext = new(11);
        }

        public ClusteringMetrics Evalute()
        {
            var testSetTransform = TrainedModel.Transform(DataSplit.TestSet);

            return MlContext.Clustering.Evaluate(
                                            data: testSetTransform,
                                            labelColumnName: "PredictedLabel",
                                            scoreColumnName: "Score",
                                            featureColumnName: "Features"
                                            );
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

        private EstimatorChain<ColumnConcatenatingTransformer> BuildDataProcessingPipeline()
        {
            var pipeline = MlContext.Transforms.Text.FeaturizeText(inputColumnName: "Sex", outputColumnName: "SexFeaturized")
                            .Append(MlContext.Transforms.Text.FeaturizeText(inputColumnName: "Island", outputColumnName: "IslandFeaturized"))
                            .Append(MlContext.Transforms.Concatenate("Features",
                                                                     "IslandFeaturized",
                                                                     nameof(PenguinData.CulmenLength),
                                                                     nameof(PenguinData.CulmenDepth),
                                                                     nameof(PenguinData.BodyMass),
                                                                     nameof(PenguinData.FilperLength),
                                                                     "SexFeaturized"))
                            .AppendCacheCheckpoint(MlContext);

            return pipeline;
        }
    }
}
