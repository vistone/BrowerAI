//! 组件构建器

use crate::*;
use anyhow::Result;

pub struct ComponentBuilder {
    config: GenerationConfig,
}

impl ComponentBuilder {
    pub fn new(config: &GenerationConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub async fn build_components(
        &self,
        analysis: &visual_learner::VisualAnalysis,
        behaviors: &[interaction_patterns::InteractionPattern],
    ) -> Result<Vec<GeneratedFile>> {
        let mut components = Vec::new();

        // 为每个检测到的组件生成代码
        for visual_comp in &analysis.components {
            let component = self.build_single_component(visual_comp, behaviors).await?;
            components.push(component);
        }

        // 生成布局组件
        let layout = self.build_layout_component(analysis).await?;
        components.push(layout);

        Ok(components)
    }

    async fn build_single_component(
        &self,
        visual_comp: &visual_learner::VisualComponent,
        behaviors: &[interaction_patterns::InteractionPattern],
    ) -> Result<GeneratedFile> {
        let component_name = format!("{:?}", visual_comp.component_type);

        let content = match self.config.target_framework {
            Framework::React => self.build_react_component(visual_comp, behaviors),
            Framework::Vue => self.build_vue_component(visual_comp, behaviors),
            Framework::Svelte => self.build_svelte_component(visual_comp, behaviors),
            Framework::VanillaJS => self.build_vanilla_component(visual_comp, behaviors),
            _ => anyhow::bail!("Framework not supported"),
        }?;

        let extension = match self.config.target_framework {
            Framework::React => "tsx",
            Framework::Vue => "vue",
            Framework::Svelte => "svelte",
            Framework::VanillaJS => "js",
            _ => "tsx",
        };

        Ok(GeneratedFile {
            path: format!("src/components/{}.{}", component_name, extension),
            content,
            file_type: FileType::Component,
        })
    }

    fn build_react_component(
        &self,
        comp: &visual_learner::VisualComponent,
        behaviors: &[interaction_patterns::InteractionPattern],
    ) -> Result<String> {
        let component_name = format!("{:?}", comp.component_type);
        let class_name = component_name.to_lowercase();

        // 查找相关的行为
        let related_behaviors: Vec<_> = behaviors
            .iter()
            .filter(|b| self.is_behavior_related(b, comp))
            .collect();

        let mut imports = vec!["import React from 'react';".to_string()];
        let mut hooks = Vec::new();
        let _event_handlers: Vec<String> = Vec::new();

        // 根据行为添加hooks和事件处理
        for behavior in related_behaviors {
            if behavior.pattern_type == interaction_patterns::ComplexPatternType::DragAndDrop {
                imports.push("import { useDragDrop } from '../hooks/useDragDrop';".to_string());
                hooks.push(format!(
                    "  const {{ {}, isDragging }} = useDragDrop();",
                    component_name.to_lowercase()
                ));
            }
        }

        // 生成样式类名
        let style_classes = self.generate_style_classes(comp);

        Ok(format!(
            r#"{}
import './{}.css';

export interface {}Props {{
  className?: string;
  children?: React.ReactNode;
  onClick?: () => void;
}}

export const {}: React.FC<{}Props> = ({{ 
  className = '',
  children,
  onClick 
}}) => {{
{}

  return (
    <div 
      className="{}"
      onClick={{onClick}}
      style={{{{ width: {}px, height: {}px }}}}
    >
      {{children}}
    </div>
  );
}};

export default {};
"#,
            imports.join("\n"),
            class_name,
            component_name,
            component_name,
            component_name,
            hooks.join("\n"),
            style_classes,
            comp.bounding_box.width,
            comp.bounding_box.height,
            component_name
        ))
    }

    fn build_vue_component(
        &self,
        comp: &visual_learner::VisualComponent,
        _behaviors: &[interaction_patterns::InteractionPattern],
    ) -> Result<String> {
        let component_name = format!("{:?}", comp.component_type);
        let class_name = component_name.to_lowercase();

        Ok(format!(
            r#"<template>
  <div 
    class="{}"
    :style="{{ width: '{}px', height: '{}px' }}"
    @click="$emit('click')"
  >
    <slot />
  </div>
</template>

<script>
export default {{
  name: '{}',
  props: {{
    className: {{
      type: String,
      default: ''
    }}
  }},
  emits: ['click']
}}
</script>

<style scoped>
@import './{}.css';
</style>
"#,
            class_name,
            comp.bounding_box.width,
            comp.bounding_box.height,
            component_name,
            class_name
        ))
    }

    fn build_svelte_component(
        &self,
        comp: &visual_learner::VisualComponent,
        _behaviors: &[interaction_patterns::InteractionPattern],
    ) -> Result<String> {
        let component_name = format!("{:?}", comp.component_type);
        let class_name = component_name.to_lowercase();

        Ok(format!(
            r#"<script>
  import {{ createEventDispatcher }} from 'svelte';
  
  export let className = '';
  
  const dispatch = createEventDispatcher();
  
  function handleClick() {{
    dispatch('click');
  }}
</script>

<div 
  class="{} {{className}}"
  style="width: {}px; height: {}px;"
  on:click={{handleClick}}
>
  <slot />
</div>

<style>
  @import './{}.css';
</style>
"#,
            class_name, comp.bounding_box.width, comp.bounding_box.height, class_name
        ))
    }

    fn build_vanilla_component(
        &self,
        comp: &visual_learner::VisualComponent,
        _behaviors: &[interaction_patterns::InteractionPattern],
    ) -> Result<String> {
        let component_name = format!("{:?}", comp.component_type);
        let class_name = component_name.to_lowercase();

        Ok(format!(
            r#"export class {} {{
  constructor(element, options = {{}}) {{
    this.element = element;
    this.options = options;
    this.init();
  }}

  init() {{
    this.element.classList.add('{}');
    this.element.style.width = '{}px';
    this.element.style.height = '{}px';
    
    this.bindEvents();
  }}

  bindEvents() {{
    this.element.addEventListener('click', () => {{
      this.options.onClick?.();
    }});
  }}

  destroy() {{
    // Cleanup
  }}
}}

export default {};
"#,
            component_name,
            class_name,
            comp.bounding_box.width,
            comp.bounding_box.height,
            component_name
        ))
    }

    fn is_behavior_related(
        &self,
        behavior: &interaction_patterns::InteractionPattern,
        component: &visual_learner::VisualComponent,
    ) -> bool {
        // 根据组件类型和行为类型判断是否相关
        match (&component.component_type, &behavior.pattern_type) {
            (visual_learner::ComponentType::Button, _) => true,
            (
                visual_learner::ComponentType::Input,
                interaction_patterns::ComplexPatternType::DragAndDrop,
            ) => false,
            _ => true,
        }
    }

    fn generate_style_classes(&self, comp: &visual_learner::VisualComponent) -> String {
        let mut classes = vec![format!("{:?}", comp.component_type).to_lowercase()];

        if comp.visual_style.border_radius > 0 {
            classes.push("rounded".to_string());
        }

        classes.join(" ")
    }

    async fn build_layout_component(
        &self,
        analysis: &visual_learner::VisualAnalysis,
    ) -> Result<GeneratedFile> {
        let content = match self.config.target_framework {
            Framework::React => self.build_react_layout(analysis),
            _ => anyhow::bail!("Framework not supported for layout"),
        }?;

        Ok(GeneratedFile {
            path: "src/components/Layout.tsx".to_string(),
            content,
            file_type: FileType::Component,
        })
    }

    fn build_react_layout(&self, analysis: &visual_learner::VisualAnalysis) -> Result<String> {
        let mut sections = String::new();

        for section in &analysis.layout.sections {
            let section_component = match section.section_type {
                visual_learner::SectionType::Header => "Header",
                visual_learner::SectionType::Content => "main",
                visual_learner::SectionType::Footer => "Footer",
                _ => "section",
            };

            sections.push_str(&format!(
                "      <{} className=\"{}\">\n        {{/* {} content */}}\n      </{}>\n",
                section_component,
                section.name.to_lowercase(),
                section.name,
                section_component
            ));
        }

        Ok(format!(
            r#"import React from 'react';
import './Layout.css';

export const Layout: React.FC = ({{ children }}) => {{
  return (
    <div className="layout layout--{}">
{}
      {{children}}
    </div>
  );
}};

export default Layout;
"#,
            format!("{:?}", analysis.layout.layout_type).to_lowercase(),
            sections
        ))
    }
}
