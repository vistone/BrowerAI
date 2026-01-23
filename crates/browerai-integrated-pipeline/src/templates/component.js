// Component utilities for BrowerAI
class ComponentManager {
    constructor() {
        this.components = new Map();
    }
    
    register(name, component) {
        this.components.set(name, component);
    }
    
    get(name) {
        return this.components.get(name);
    }
    
    initialize(element) {
        const name = element.dataset.component;
        const Component = this.get(name);
        if (Component) {
            return new Component(element);
        }
        return null;
    }
}

window.ComponentManager = new ComponentManager();
