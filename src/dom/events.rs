use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Event types supported by the DOM
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EventType {
    // Mouse events
    Click,
    MouseDown,
    MouseUp,
    MouseMove,
    MouseEnter,
    MouseLeave,
    
    // Keyboard events
    KeyDown,
    KeyUp,
    KeyPress,
    
    // Focus events
    Focus,
    Blur,
    
    // Form events
    Submit,
    Change,
    Input,
    
    // Load events
    Load,
    Unload,
    
    // Custom event
    Custom(String),
}

/// Event phase during propagation
#[derive(Debug, Clone, PartialEq)]
pub enum EventPhase {
    None,
    Capturing,
    AtTarget,
    Bubbling,
}

/// DOM Event object
#[derive(Clone)]
pub struct Event {
    pub event_type: EventType,
    pub target: Arc<RwLock<super::DomNode>>,
    pub timestamp: u64,
    pub phase: EventPhase,
    pub bubbles: bool,
    pub cancelable: bool,
    propagation_stopped: Arc<RwLock<bool>>,
    default_prevented: Arc<RwLock<bool>>,
}

impl Event {
    /// Create a new event
    pub fn new(event_type: EventType, target: Arc<RwLock<super::DomNode>>) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        Self {
            event_type,
            target,
            timestamp,
            phase: EventPhase::None,
            bubbles: true,
            cancelable: true,
            propagation_stopped: Arc::new(RwLock::new(false)),
            default_prevented: Arc::new(RwLock::new(false)),
        }
    }
    
    /// Stop event propagation
    pub fn stop_propagation(&self) {
        *self.propagation_stopped.write().unwrap() = true;
    }
    
    /// Check if propagation was stopped
    pub fn is_propagation_stopped(&self) -> bool {
        *self.propagation_stopped.read().unwrap()
    }
    
    /// Prevent default behavior
    pub fn prevent_default(&self) {
        if self.cancelable {
            *self.default_prevented.write().unwrap() = true;
        }
    }
    
    /// Check if default was prevented
    pub fn is_default_prevented(&self) -> bool {
        *self.default_prevented.read().unwrap()
    }
}

/// Event listener function type
pub type EventListener = Box<dyn Fn(&Event) + Send + Sync>;

/// Event listener manager
pub struct EventListeners {
    listeners: HashMap<EventType, Vec<EventListener>>,
}

impl EventListeners {
    /// Create new event listeners manager
    pub fn new() -> Self {
        Self {
            listeners: HashMap::new(),
        }
    }
    
    /// Add an event listener
    pub fn add_listener<F>(&mut self, event_type: EventType, listener: F)
    where
        F: Fn(&Event) + Send + Sync + 'static,
    {
        self.listeners
            .entry(event_type)
            .or_insert_with(Vec::new)
            .push(Box::new(listener));
    }
    
    /// Remove all listeners for an event type
    pub fn remove_listeners(&mut self, event_type: &EventType) {
        self.listeners.remove(event_type);
    }
    
    /// Dispatch an event to all listeners
    pub fn dispatch(&self, event: &Event) {
        if let Some(listeners) = self.listeners.get(&event.event_type) {
            for listener in listeners {
                if event.is_propagation_stopped() {
                    break;
                }
                listener(event);
            }
        }
    }
}

impl Default for EventListeners {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dom::{DomNode, Document};
    
    #[test]
    fn test_event_creation() {
        let doc = Document::new();
        let element = doc.create_element("div");
        
        let event = Event::new(EventType::Click, element.clone());
        
        assert_eq!(event.event_type, EventType::Click);
        assert!(event.bubbles);
        assert!(event.cancelable);
        assert!(!event.is_propagation_stopped());
        assert!(!event.is_default_prevented());
    }
    
    #[test]
    fn test_event_listener() {
        let mut listeners = EventListeners::new();
        let called = Arc::new(RwLock::new(false));
        let called_clone = called.clone();
        
        listeners.add_listener(EventType::Click, move |_event| {
            *called_clone.write().unwrap() = true;
        });
        
        let doc = Document::new();
        let element = doc.create_element("div");
        let event = Event::new(EventType::Click, element);
        
        listeners.dispatch(&event);
        
        assert!(*called.read().unwrap());
    }
    
    #[test]
    fn test_event_dispatch() {
        let mut listeners = EventListeners::new();
        let counter = Arc::new(RwLock::new(0));
        let counter_clone = counter.clone();
        
        listeners.add_listener(EventType::Click, move |_event| {
            *counter_clone.write().unwrap() += 1;
        });
        
        let doc = Document::new();
        let element = doc.create_element("div");
        let event = Event::new(EventType::Click, element);
        
        listeners.dispatch(&event);
        listeners.dispatch(&event);
        
        assert_eq!(*counter.read().unwrap(), 2);
    }
    
    #[test]
    fn test_stop_propagation() {
        let mut listeners = EventListeners::new();
        let counter = Arc::new(RwLock::new(0));
        
        let counter1 = counter.clone();
        listeners.add_listener(EventType::Click, move |event| {
            *counter1.write().unwrap() += 1;
            event.stop_propagation();
        });
        
        let counter2 = counter.clone();
        listeners.add_listener(EventType::Click, move |_event| {
            *counter2.write().unwrap() += 1;
        });
        
        let doc = Document::new();
        let element = doc.create_element("div");
        let event = Event::new(EventType::Click, element);
        
        listeners.dispatch(&event);
        
        // Only first listener should execute
        assert_eq!(*counter.read().unwrap(), 1);
        assert!(event.is_propagation_stopped());
    }
    
    #[test]
    fn test_prevent_default() {
        let doc = Document::new();
        let element = doc.create_element("a");
        let event = Event::new(EventType::Click, element);
        
        assert!(!event.is_default_prevented());
        event.prevent_default();
        assert!(event.is_default_prevented());
    }
    
    #[test]
    fn test_event_propagation() {
        let mut listeners = EventListeners::new();
        let events = Arc::new(RwLock::new(Vec::new()));
        
        let events_clone = events.clone();
        listeners.add_listener(EventType::Click, move |event| {
            events_clone.write().unwrap().push(format!("{:?}", event.phase));
        });
        
        let doc = Document::new();
        let element = doc.create_element("button");
        
        // Simulate capture phase
        let mut event = Event::new(EventType::Click, element.clone());
        event.phase = EventPhase::Capturing;
        listeners.dispatch(&event);
        
        // Simulate target phase
        event.phase = EventPhase::AtTarget;
        listeners.dispatch(&event);
        
        // Simulate bubble phase
        event.phase = EventPhase::Bubbling;
        listeners.dispatch(&event);
        
        let recorded = events.read().unwrap();
        assert_eq!(recorded.len(), 3);
        assert!(recorded[0].contains("Capturing"));
        assert!(recorded[1].contains("AtTarget"));
        assert!(recorded[2].contains("Bubbling"));
    }
}
