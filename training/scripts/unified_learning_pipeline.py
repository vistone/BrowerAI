#!/usr/bin/env python3
"""
统一学习管道 - Week 6 完整集成系统
===================================

这是一个完整的学习管道，集成了所有组件:
1. ✅ 真实数据采集 (GitHub + 本地项目)
2. ✅ 数据增强 (多种变换方法)
3. ✅ 高级混淆生成 (12+ 技术)
4. ✅ 混淆效果评估
5. ✅ GPU加速训练 (PyTorch)
6. ✅ 性能指标记录

使用流程:
  python unified_learning_pipeline.py --mode full
  python unified_learning_pipeline.py --mode collect
  python unified_learning_pipeline.py --mode train --gpu cuda:0
"""

import sys
import os
import logging
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# 添加脚本目录到路径
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedLearningPipeline:
    """统一学习管道 - 集成所有模块"""
    
    def __init__(self, output_base_dir: str = 'data/week6_unified_learning'):
        self.output_dir = Path(output_base_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_log = {
            'start_time': datetime.now().isoformat(),
            'stages': {},
        }
    
    def stage_header(self, stage_name: str, description: str):
        """打印阶段标头"""
        logger.info("\n" + "="*80)
        logger.info(f"阶段: {stage_name}")
        logger.info(f"描述: {description}")
        logger.info("="*80)
    
    def run_data_collection(self):
        """阶段 1: 数据采集"""
        self.stage_header(
            "数据采集",
            "从GitHub和本地项目采集真实JavaScript代码"
        )
        
        try:
            from real_data_learning_pipeline import RealDataCollector
            
            collector = RealDataCollector(
                output_dir=str(self.output_dir / 'raw_data')
            )
            
            # 采集GitHub框架代码
            logger.info("\n📥 从GitHub采集框架代码...")
            repos = [
                {'owner': 'facebook', 'name': 'react'},
                {'owner': 'vuejs', 'name': 'vue'},
                {'owner': 'angular', 'name': 'angular'},
                {'owner': 'vercel', 'name': 'next.js'},
            ]
            
            try:
                collector.collect_from_github(repos, max_files_per_repo=30)
                logger.info(f"✅ GitHub采集完成")
            except Exception as e:
                logger.warning(f"⚠️  GitHub采集失败: {e}")
            
            # 采集本地项目
            logger.info("\n📁 从本地项目采集代码...")
            local_paths = [
                'training/real_data',
                'crates/browerai-core',
                'crates/browerai-renderer',
            ]
            
            collector.collect_from_local_projects(local_paths)
            
            # 保存采集的数据
            output_file = collector.save_collected_data()
            
            self.pipeline_log['stages']['data_collection'] = {
                'status': 'completed',
                'samples_collected': len(collector.collected_samples),
                'output': str(output_file),
                'timestamp': datetime.now().isoformat(),
            }
            
            return collector.collected_samples
            
        except Exception as e:
            logger.error(f"❌ 数据采集失败: {e}")
            self.pipeline_log['stages']['data_collection'] = {
                'status': 'failed',
                'error': str(e),
            }
            return []
    
    def run_data_augmentation(self, samples: List[Dict]):
        """阶段 2: 数据增强"""
        self.stage_header(
            "数据增强",
            "对真实代码进行多种变换以增加样本多样性"
        )
        
        try:
            from real_data_learning_pipeline import DataAugmentation
            
            augmenter = DataAugmentation()
            
            logger.info(f"\n🔄 对 {len(samples)} 个样本进行增强...")
            augmented_samples = augmenter.augment_dataset(
                samples,
                augmentation_factor=3
            )
            
            self.pipeline_log['stages']['data_augmentation'] = {
                'status': 'completed',
                'original_samples': len(samples),
                'augmented_samples': len(augmented_samples),
                'augmentation_factor': 3,
                'timestamp': datetime.now().isoformat(),
            }
            
            return augmented_samples
            
        except Exception as e:
            logger.error(f"❌ 数据增强失败: {e}")
            self.pipeline_log['stages']['data_augmentation'] = {
                'status': 'failed',
                'error': str(e),
            }
            return samples
    
    def run_obfuscation_generation(self):
        """阶段 3: 混淆样本生成"""
        self.stage_header(
            "混淆样本生成",
            "使用12+种混淆技术生成大规模训练数据"
        )
        
        try:
            from advanced_obfuscation_generator import AdvancedObfuscationGenerator
            
            generator = AdvancedObfuscationGenerator(
                output_dir=str(self.output_dir / 'obfuscation_samples')
            )
            
            logger.info(f"\n🔀 生成混淆样本...")
            generator.generate_samples(
                num_samples=500,
                num_techniques=3
            )
            
            generator.save_samples()
            
            self.pipeline_log['stages']['obfuscation_generation'] = {
                'status': 'completed',
                'num_samples': len(generator.generated_samples),
                'techniques': sorted(generator.technique_stats.keys()),
                'timestamp': datetime.now().isoformat(),
            }
            
            return generator.generated_samples
            
        except Exception as e:
            logger.error(f"❌ 混淆生成失败: {e}")
            self.pipeline_log['stages']['obfuscation_generation'] = {
                'status': 'failed',
                'error': str(e),
            }
            return []
    
    def run_obfuscation_evaluation(self, samples: List[Dict]):
        """阶段 4: 混淆效果评估"""
        self.stage_header(
            "混淆评估",
            "评估各种混淆技术的效果"
        )
        
        try:
            from real_data_learning_pipeline import ObfuscationEvaluator
            
            evaluator = ObfuscationEvaluator()
            
            # 按技术分组
            techniques = {}
            for sample in samples:
                technique = sample.get('technique', 'unknown')
                if technique not in techniques:
                    techniques[technique] = []
                techniques[technique].append(sample)
            
            logger.info(f"\n📊 评估 {len(techniques)} 种混淆技术...")
            
            evaluation_results = {}
            for technique, technique_samples in techniques.items():
                result = evaluator.evaluate_obfuscation_technique(
                    technique,
                    technique_samples
                )
                if result:
                    evaluation_results[technique] = result
            
            # 保存评估结果
            eval_file = self.output_dir / 'evaluation_results.json'
            with open(eval_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            
            logger.info(f"💾 评估结果保存到: {eval_file}")
            
            self.pipeline_log['stages']['obfuscation_evaluation'] = {
                'status': 'completed',
                'techniques_evaluated': len(evaluation_results),
                'timestamp': datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"❌ 混淆评估失败: {e}")
            self.pipeline_log['stages']['obfuscation_evaluation'] = {
                'status': 'failed',
                'error': str(e),
            }
    
    def run_gpu_training(self):
        """阶段 5: GPU加速训练"""
        self.stage_header(
            "GPU训练",
            "使用GPU进行深度学习模型训练"
        )
        
        try:
            from gpu_unified_training import TrainingPipeline
            
            pipeline = TrainingPipeline(
                output_dir=str(self.output_dir / 'gpu_training')
            )
            
            logger.info(f"\n🚀 启动GPU训练...")
            pipeline.run(
                num_samples=1000,
                batch_size=64,
                epochs=100
            )
            
            self.pipeline_log['stages']['gpu_training'] = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"❌ GPU训练失败: {e}")
            self.pipeline_log['stages']['gpu_training'] = {
                'status': 'failed',
                'error': str(e),
            }
    
    def run_full_pipeline(self):
        """运行完整管道"""
        logger.info("""
╔════════════════════════════════════════════════════════════════════════════════╗
║               统一学习管道 - Week 6 完整集成系统                              ║
║                                                                                ║
║  流程:                                                                         ║
║  1️⃣  真实数据采集 (GitHub + 本地项目)                                        ║
║  2️⃣  数据增强 (多种代码变换)                                                 ║
║  3️⃣  混淆样本生成 (12+ 技术)                                                 ║
║  4️⃣  混淆效果评估                                                             ║
║  5️⃣  GPU加速训练 (PyTorch)                                                   ║
║  6️⃣  性能指标记录                                                             ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
""")
        
        # 阶段 1: 数据采集
        collected_samples = self.run_data_collection()
        
        # 阶段 2: 数据增强
        augmented_samples = self.run_data_augmentation(collected_samples)
        
        # 阶段 3: 混淆生成
        obfuscation_samples = self.run_obfuscation_generation()
        
        # 阶段 4: 评估
        self.run_obfuscation_evaluation(obfuscation_samples)
        
        # 阶段 5: 训练
        self.run_gpu_training()
        
        # 保存管道日志
        self.pipeline_log['end_time'] = datetime.now().isoformat()
        log_file = self.output_dir / 'pipeline_log.json'
        with open(log_file, 'w') as f:
            json.dump(self.pipeline_log, f, indent=2)
        
        logger.info(f"\n💾 管道日志保存到: {log_file}")
    
    def run_partial_pipeline(self, stages: List[str]):
        """运行指定的管道阶段"""
        stage_map = {
            'collect': self.run_data_collection,
            'augment': self.run_data_augmentation,
            'generate': self.run_obfuscation_generation,
            'evaluate': self.run_obfuscation_evaluation,
            'train': self.run_gpu_training,
        }
        
        for stage in stages:
            if stage in stage_map:
                if stage == 'augment':
                    samples = self.run_data_collection()
                    stage_map[stage](samples)
                elif stage == 'evaluate':
                    samples = self.run_obfuscation_generation()
                    stage_map[stage](samples)
                else:
                    stage_map[stage]()
            else:
                logger.warning(f"⚠️  未知阶段: {stage}")


def main():
    parser = argparse.ArgumentParser(
        description='统一学习管道 - Week 6 完整集成系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行完整管道
  python unified_learning_pipeline.py --mode full
  
  # 仅采集数据
  python unified_learning_pipeline.py --mode collect
  
  # 运行特定阶段
  python unified_learning_pipeline.py --stages collect generate train
  
  # 使用特定GPU
  python unified_learning_pipeline.py --mode full --gpu cuda:0
"""
    )
    
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'collect', 'augment', 'generate', 'evaluate', 'train'],
                       help='执行模式')
    parser.add_argument('--stages', nargs='+', default=[],
                       help='指定执行的阶段 (collect, augment, generate, evaluate, train)')
    parser.add_argument('--output', type=str, default='data/week6_unified_learning',
                       help='输出目录')
    parser.add_argument('--gpu', type=str, default='auto',
                       help='GPU设备 (cuda:0, cuda:1, auto, cpu)')
    parser.add_argument('--samples', type=int, default=500,
                       help='混淆样本数')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    
    args = parser.parse_args()
    
    # 初始化管道
    pipeline = UnifiedLearningPipeline(output_base_dir=args.output)
    
    # 设置GPU环境变量
    if args.gpu != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu.split(':')[1] if ':' in args.gpu else '0'
    
    # 执行管道
    if args.mode == 'full':
        pipeline.run_full_pipeline()
    elif args.stages:
        pipeline.run_partial_pipeline(args.stages)
    else:
        mode_map = {
            'collect': pipeline.run_data_collection,
            'augment': lambda: pipeline.run_data_augmentation(pipeline.run_data_collection()),
            'generate': pipeline.run_obfuscation_generation,
            'evaluate': lambda: pipeline.run_obfuscation_evaluation(pipeline.run_obfuscation_generation()),
            'train': pipeline.run_gpu_training,
        }
        
        if args.mode in mode_map:
            mode_map[args.mode]()
        else:
            logger.error(f"❌ 未知模式: {args.mode}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ 学习管道执行完成!")
    logger.info(f"📁 结果保存到: {pipeline.output_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
