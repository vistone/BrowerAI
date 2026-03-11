//! 测试场景定义

use serde::{Deserialize, Serialize};

/// 测试场景
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestScenario {
    pub name: String,
    pub description: String,
    pub steps: Vec<TestStep>,
}

/// 测试步骤
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestStep {
    pub name: String,
    pub action: TestAction,
    pub assertions: Vec<TestAssertion>,
}

/// 测试动作
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestAction {
    Click(String), // selector
    Type {
        selector: String,
        value: String,
    },
    Scroll {
        direction: ScrollDirection,
        amount: u32,
    },
    Wait(u64),             // milliseconds
    Screenshot(String),    // name
    AssertVisible(String), // selector
    AssertText {
        selector: String,
        expected: String,
    },
    AssertUrl(String),
    KeyPress(String), // key
}

/// 滚动方向
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScrollDirection {
    Up,
    Down,
    Left,
    Right,
}

/// 测试断言
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestAssertion {
    pub assertion_type: AssertionType,
    pub selector: Option<String>,
    pub expected: serde_json::Value,
}

/// 断言类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AssertionType {
    ElementExists,
    ElementVisible,
    ElementTextEquals,
    ElementTextContains,
    UrlEquals,
    UrlContains,
    Custom(String),
}

/// 预定义测试场景
pub struct PredefinedScenarios;

impl PredefinedScenarios {
    /// 导航测试
    pub fn navigation() -> TestScenario {
        TestScenario {
            name: "Navigation".to_string(),
            description: "Test basic navigation functionality".to_string(),
            steps: vec![
                TestStep {
                    name: "Click navigation link".to_string(),
                    action: TestAction::Click("nav a:first-child".to_string()),
                    assertions: vec![],
                },
                TestStep {
                    name: "Wait for navigation".to_string(),
                    action: TestAction::Wait(500),
                    assertions: vec![],
                },
                TestStep {
                    name: "Assert URL changed".to_string(),
                    action: TestAction::AssertUrl("/".to_string()),
                    assertions: vec![],
                },
            ],
        }
    }

    /// 表单测试
    pub fn form_submission() -> TestScenario {
        TestScenario {
            name: "Form Submission".to_string(),
            description: "Test form input and submission".to_string(),
            steps: vec![
                TestStep {
                    name: "Fill email field".to_string(),
                    action: TestAction::Type {
                        selector: "input[type='email']".to_string(),
                        value: "test@example.com".to_string(),
                    },
                    assertions: vec![],
                },
                TestStep {
                    name: "Fill password field".to_string(),
                    action: TestAction::Type {
                        selector: "input[type='password']".to_string(),
                        value: "password123".to_string(),
                    },
                    assertions: vec![],
                },
                TestStep {
                    name: "Click submit button".to_string(),
                    action: TestAction::Click("button[type='submit']".to_string()),
                    assertions: vec![],
                },
                TestStep {
                    name: "Wait for response".to_string(),
                    action: TestAction::Wait(1000),
                    assertions: vec![],
                },
            ],
        }
    }

    /// 搜索测试
    pub fn search() -> TestScenario {
        TestScenario {
            name: "Search".to_string(),
            description: "Test search functionality".to_string(),
            steps: vec![
                TestStep {
                    name: "Focus search input".to_string(),
                    action: TestAction::Click("input[type='search']".to_string()),
                    assertions: vec![],
                },
                TestStep {
                    name: "Type search query".to_string(),
                    action: TestAction::Type {
                        selector: "input[type='search']".to_string(),
                        value: "test query".to_string(),
                    },
                    assertions: vec![],
                },
                TestStep {
                    name: "Wait for debounce".to_string(),
                    action: TestAction::Wait(300),
                    assertions: vec![],
                },
                TestStep {
                    name: "Press Enter".to_string(),
                    action: TestAction::KeyPress("Enter".to_string()),
                    assertions: vec![],
                },
                TestStep {
                    name: "Wait for results".to_string(),
                    action: TestAction::Wait(500),
                    assertions: vec![],
                },
            ],
        }
    }

    /// 滚动测试
    pub fn infinite_scroll() -> TestScenario {
        TestScenario {
            name: "Infinite Scroll".to_string(),
            description: "Test infinite scroll functionality".to_string(),
            steps: vec![
                TestStep {
                    name: "Scroll down".to_string(),
                    action: TestAction::Scroll {
                        direction: ScrollDirection::Down,
                        amount: 1000,
                    },
                    assertions: vec![],
                },
                TestStep {
                    name: "Wait for load".to_string(),
                    action: TestAction::Wait(1000),
                    assertions: vec![],
                },
                TestStep {
                    name: "Scroll down again".to_string(),
                    action: TestAction::Scroll {
                        direction: ScrollDirection::Down,
                        amount: 1000,
                    },
                    assertions: vec![],
                },
                TestStep {
                    name: "Wait for load".to_string(),
                    action: TestAction::Wait(1000),
                    assertions: vec![],
                },
            ],
        }
    }

    /// 模态框测试
    pub fn modal() -> TestScenario {
        TestScenario {
            name: "Modal".to_string(),
            description: "Test modal open and close".to_string(),
            steps: vec![
                TestStep {
                    name: "Click to open modal".to_string(),
                    action: TestAction::Click("[data-open-modal]".to_string()),
                    assertions: vec![],
                },
                TestStep {
                    name: "Wait for animation".to_string(),
                    action: TestAction::Wait(300),
                    assertions: vec![],
                },
                TestStep {
                    name: "Assert modal visible".to_string(),
                    action: TestAction::AssertVisible(".modal".to_string()),
                    assertions: vec![],
                },
                TestStep {
                    name: "Click to close modal".to_string(),
                    action: TestAction::Click("[data-close-modal]".to_string()),
                    assertions: vec![],
                },
                TestStep {
                    name: "Wait for animation".to_string(),
                    action: TestAction::Wait(300),
                    assertions: vec![],
                },
            ],
        }
    }

    /// 获取所有预定义场景
    pub fn all() -> Vec<TestScenario> {
        vec![
            Self::navigation(),
            Self::form_submission(),
            Self::search(),
            Self::infinite_scroll(),
            Self::modal(),
        ]
    }
}
