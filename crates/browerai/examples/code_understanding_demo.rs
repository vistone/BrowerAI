//! 代码理解系统演示
//!
//! 展示如何分析开源项目的架构、模块结构和数据流
//!
//! 运行: cargo run --example code_understanding_demo

use browerai_learning::CodeUnderstandingSystem;
use std::fs;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    println!("🔍 代码理解系统 - 架构分析演示");
    println!("================================\n");

    // 示例1: 分析简单的库代码
    analyze_simple_library()?;

    println!("\n");

    // 示例2: 分析模块化代码
    analyze_modular_code()?;

    println!("\n");

    // 示例3: 分析混淆后的代码
    analyze_obfuscated_code()?;

    Ok(())
}

fn analyze_simple_library() -> anyhow::Result<()> {
    println!("📚 示例1: 分析简单库代码\n");

    let code = r#"
        // 日期处理库
        export function parseDate(dateStr) {
            return new Date(dateStr);
        }

        export function formatDate(date, format) {
            // 格式化日期
            return date.toLocaleDateString();
        }

        export function addDays(date, days) {
            date.setDate(date.getDate() + days);
            return date;
        }

        export class DateRange {
            constructor(start, end) {
                this.start = start;
                this.end = end;
            }

            getDays() {
                return Math.floor((this.end - this.start) / (1000 * 60 * 60 * 24));
            }
        }
    "#;

    let system = CodeUnderstandingSystem::new();
    let report = system.analyze(code, "DateLib v1.0")?;

    // 输出架构信息
    println!("✅ 架构检测结果:");
    println!("   模式: {:?}", report.architecture.pattern);
    println!(
        "   特征: {}\n",
        report.architecture.characteristics.join(", ")
    );

    // 输出模块信息
    println!("✅ 模块结构:");
    for module in &report.modules {
        println!("   📦 {}", module.name);
        println!("      职责: {}", module.responsibility);
        if !module.exports.is_empty() {
            println!("      导出: {}", module.exports.join(", "));
        }
    }

    // 输出 API
    println!("\n✅ 公共 API:");
    for api in report.apis.iter().take(5) {
        println!("   - {}", api.signature);
    }

    // 生成完整文本报告
    let report_text = system.generate_report(&report);
    println!("\n{}", report_text);

    // 生成 Mermaid 图表
    let mermaid = system.visualize(&report, browerai_learning::VisualizationFormat::Mermaid)?;
    println!("📊 Mermaid 图表:\n{}", mermaid);

    Ok(())
}

fn analyze_modular_code() -> anyhow::Result<()> {
    println!("📦 示例2: 分析模块化代码\n");

    let code = r#"
        // 用户服务模块
        import { Database } from './database.js';
        import { Logger } from './logger.js';

        export class UserService {
            constructor() {
                this.db = new Database();
                this.logger = new Logger();
            }

            async getUserById(id) {
                this.logger.debug(`Fetching user ${id}`);
                const user = await this.db.query('SELECT * FROM users WHERE id = ?', [id]);
                return user;
            }

            async createUser(userData) {
                const userId = await this.db.insert('users', userData);
                this.logger.info(`User created: ${userId}`);
                return userId;
            }

            async updateUser(id, updates) {
                await this.db.update('users', updates, { id });
                this.logger.info(`User ${id} updated`);
            }
        }

        export async function initializeService() {
            return new UserService();
        }
    "#;

    let system = CodeUnderstandingSystem::new();
    let report = system.analyze(code, "UserService v2.1")?;

    println!("✅ 架构: {:?}", report.architecture.pattern);
    println!("✅ 发现 {} 个模块", report.modules.len());
    println!("✅ 识别到 {} 条数据流", report.dataflows.len());
    println!("✅ 导出 {} 个公共 API\n", report.apis.len());

    // 显示依赖关系
    println!("🔗 依赖关系:");
    for module in &report.modules {
        if !module.dependencies.is_empty() {
            println!("   {} → {}", module.name, module.dependencies.join(", "));
        }
    }

    Ok(())
}

fn analyze_obfuscated_code() -> anyhow::Result<()> {
    println!("🔐 示例3: 分析混淆代码\n");

    let code = r#"
        !function(a,b){var c=function(){this.d=a(b)};c.prototype.e=function(){return this.d.f()};module.exports=c}(require('./x'),require('./y'));
    "#;

    let system = CodeUnderstandingSystem::new();
    let report = system.analyze(code, "ObfuscatedLib")?;

    println!("✅ 架构: {:?}", report.architecture.pattern);
    println!("✅ 代码复杂度: {}", report.statistics.complexity_level);
    println!("✅ 函数数量: {}", report.statistics.function_count);
    println!("✅ 变量数量: {}", report.statistics.variable_count);

    // 输出统计信息
    println!("\n📊 代码统计:");
    println!("   行数: {}", report.statistics.line_count);
    println!("   函数: {}", report.statistics.function_count);
    println!("   变量: {}", report.statistics.variable_count);
    println!("   类: {}", report.statistics.class_count);
    println!("   模块: {}", report.statistics.module_count);

    Ok(())
}
