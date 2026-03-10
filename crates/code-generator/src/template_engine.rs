//! 模板引擎

use crate::*;
use anyhow::Result;
use handlebars::Handlebars;
use std::collections::HashMap;

pub struct TemplateEngine {
    handlebars: Handlebars<'static>,
    config: GenerationConfig,
}

impl TemplateEngine {
    pub fn new(config: &GenerationConfig) -> Self {
        let mut handlebars = Handlebars::new();
        
        // 注册内置模板
        Self::register_templates(&mut handlebars);

        Self {
            handlebars,
            config: config.clone(),
        }
    }

    fn register_templates(handlebars: &mut Handlebars) {
        // 组件模板
        handlebars.register_template_string("react_component", r#"
import React from 'react';
import './{{name}}.css';

export interface {{name}}Props {
{{#each props}}
  {{name}}{{#if optional}}?{{/if}}: {{type}};
{{/each}}
}

export const {{name}}: React.FC<{{name}}Props> = ({
{{#each props}}
  {{name}}{{#if default}} = {{default}}{{/if}},
{{/each}}
}) => {
  return (
    <div className="{{class_name}}"{{#if aria_label}} aria-label="{{aria_label}}"{{/if}}>
      {{content}}
    </div>
  );
};
"#).unwrap();

        // 样式模板
        handlebars.register_template_string("css_module", r#"
.{{class_name}} {
{{#each styles}}
  {{property}}: {{value}};
{{/each}}
}

{{#each states}}
.{{../class_name}}:{{state}} {
{{#each styles}}
  {{property}}: {{value}};
{{/each}}
}
{{/each}}

{{#each variants}}
.{{../class_name}}.{{variant}} {
{{#each styles}}
  {{property}}: {{value}};
{{/each}}
}
{{/each}}
"#).unwrap();

        // Vue组件模板
        handlebars.register_template_string("vue_component", r#"
<template>
  <div class="{{class_name}}"{{#if aria_label}} :aria-label="ariaLabel"{{/if}}>
    {{content}}
  </div>
</template>

<script>
export default {
  name: '{{name}}',
  props: {
{{#each props}}
    {{name}}: {
      type: {{type}},
      {{#if required}}required: true,{{/if}}
      {{#if default}}default: {{default}}{{/if}}
    },
{{/each}}
  }
}
</script>

<style scoped>
@import './{{name}}.css';
</style>
"#).unwrap();
    }

    pub fn render_component(&self, template_name: &str, data: &HashMap<String, serde_json::Value>) -> Result<String> {
        Ok(self.handlebars.render(template_name, data)?)
    }

    pub fn render_with_data<T: serde::Serialize>(&self, template_name: &str, data: &T) -> Result<String> {
        Ok(self.handlebars.render(template_name, data)?)
    }
}
