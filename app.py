#!/usr/bin/env python3

from aws_cdk import core
from infra.regml_stack import RegMLStack

app = core.App()
RegMLStack(app, "regml-stack") 
app.synth()
