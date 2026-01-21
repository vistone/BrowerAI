//! Data Models Module
//!
//! Central module file that re-exports all data models.

pub mod behavior_record;
pub mod page_content;

pub use page_content::{
    FormInfo, InlineContent, InputInfo, NavInfo, NavItem, PageContent, PageMetadata, Resource,
    ResourceType, SimplifiedDom, SimplifiedNode,
};

pub use behavior_record::{
    ApiCallRecord, AttributeChange, BehaviorRecord, BehaviorSummary, ChangeType, CodeLocation,
    ConsoleLog, DomMutation, DomMutationType, EventFlowRecord, EventHandlerInfo, ExecutionSnapshot,
    NetworkRequestRecord, StackFrame, StateChangeRecord,
};
