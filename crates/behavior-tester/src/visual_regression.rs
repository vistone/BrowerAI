//! 视觉回归测试器

use crate::*;
use anyhow::Result;
use image::{DynamicImage, GenericImage, GenericImageView};
use playwright::Playwright;
use std::path::PathBuf;

pub struct VisualRegressionTester;

impl VisualRegressionTester {
    pub fn new() -> Self {
        Self
    }

    pub async fn compare_pages(
        &self,
        original_url: &str,
        generated_url: &str,
    ) -> Result<TestResult> {
        let start_time = std::time::Instant::now();

        // 截图
        let original_screenshot = self.capture_screenshot(original_url).await?;
        let generated_screenshot = self.capture_screenshot(generated_url).await?;

        // 比较图片
        let comparison = self.compare_images(&original_screenshot, &generated_screenshot)?;

        let duration = start_time.elapsed().as_millis() as u64;

        Ok(TestResult {
            test_name: "Visual Regression".to_string(),
            test_type: TestType::Visual,
            passed: comparison.passed,
            score: comparison.similarity,
            duration_ms: duration,
            details: TestDetails {
                steps_executed: 1,
                assertions_passed: if comparison.passed { 1 } else { 0 },
                assertions_failed: if comparison.passed { 0 } else { 1 },
                screenshots: vec![
                    original_screenshot,
                    generated_screenshot,
                    comparison.diff_image_path.unwrap_or_default(),
                ],
                logs: vec![format!(
                    "Pixel difference: {} ({:.2}%)",
                    comparison.pixel_diff_count, comparison.pixel_diff_percentage
                )],
            },
            errors: if comparison.passed {
                vec![]
            } else {
                vec![TestError {
                    step: 0,
                    message: format!(
                        "Visual difference detected: {:.1}%",
                        (1.0 - comparison.similarity) * 100.0
                    ),
                    expected: "Identical".to_string(),
                    actual: format!("{:.1}% different", (1.0 - comparison.similarity) * 100.0),
                    severity: ErrorSeverity::Warning,
                }]
            },
        })
    }

    async fn capture_screenshot(&self, url: &str) -> Result<String> {
        let playwright = Playwright::initialize().await?;
        let browser = playwright
            .chromium()
            .launcher()
            .headless(true)
            .launch()
            .await?;
        let context = browser
            .context_builder()
            .viewport(Some(playwright::api::Viewport {
                width: 1280,
                height: 720,
            }))
            .build()
            .await?;
        let page = context.new_page().await?;

        page.goto_builder(url).goto().await?;
        // Use a simple delay instead of wait_for_load_state which doesn't exist
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

        // 等待动画完成
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        let filename = format!("screenshot_{}.png", url.replace(['/', ':', '.'], "_"));

        page.screenshot_builder()
            .path(PathBuf::from(&filename))
            .full_page(true)
            .screenshot()
            .await?;

        browser.close().await?;

        Ok(filename)
    }

    fn compare_images(&self, img1_path: &str, img2_path: &str) -> Result<VisualRegressionResult> {
        let img1 = image::open(img1_path)?;
        let img2 = image::open(img2_path)?;

        // 确保尺寸相同
        if img1.dimensions() != img2.dimensions() {
            return Ok(VisualRegressionResult {
                similarity: 0.0,
                diff_image_path: None,
                pixel_diff_count: (img1.width() * img1.height()) as usize,
                pixel_diff_percentage: 100.0,
                passed: false,
            });
        }

        let (width, height) = img1.dimensions();
        let mut diff_count = 0;
        let total_pixels = (width * height) as usize;

        // 创建差异图
        let mut diff_image = DynamicImage::new_rgba8(width, height);

        // 比较每个像素
        for y in 0..height {
            for x in 0..width {
                let p1 = img1.get_pixel(x, y);
                let p2 = img2.get_pixel(x, y);

                // 计算像素差异
                let diff = self.pixel_diff(&p1, &p2);

                if diff > 10.0 {
                    diff_count += 1;
                    // 标记差异像素为红色
                    unsafe {
                        diff_image.unsafe_put_pixel(x, y, image::Rgba([255, 0, 0, 255]));
                    }
                } else {
                    // 相似像素保持原样（半透明）
                    unsafe {
                        diff_image.unsafe_put_pixel(x, y, image::Rgba([p1[0], p1[1], p1[2], 128]));
                    }
                }
            }
        }

        // 保存差异图
        let diff_path = format!("diff_{}_{}", img1_path, img2_path);
        diff_image.save(&diff_path)?;

        let diff_percentage = (diff_count as f64 / total_pixels as f64) * 100.0;
        let similarity = 1.0 - (diff_count as f64 / total_pixels as f64);

        Ok(VisualRegressionResult {
            similarity,
            diff_image_path: Some(diff_path),
            pixel_diff_count: diff_count,
            pixel_diff_percentage: diff_percentage,
            passed: similarity >= 0.95, // 95%相似度通过
        })
    }

    fn pixel_diff(&self, p1: &image::Rgba<u8>, p2: &image::Rgba<u8>) -> f64 {
        let dr = (p1[0] as i32 - p2[0] as i32).abs() as f64;
        let dg = (p1[1] as i32 - p2[1] as i32).abs() as f64;
        let db = (p1[2] as i32 - p2[2] as i32).abs() as f64;
        let da = (p1[3] as i32 - p2[3] as i32).abs() as f64;

        (dr + dg + db + da) / 4.0
    }
}

impl Default for VisualRegressionTester {
    fn default() -> Self {
        Self::new()
    }
}
