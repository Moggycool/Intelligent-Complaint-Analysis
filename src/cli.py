#!/usr/bin/env python3
"""
CLI interface for the Complaint RAG Agent
"""
import click
import sys
import os
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_agent.agent import ComplaintRAGAgent
from rag_agent.config import Config


@click.group()
def cli():
    """Intelligent Complaint Analysis - RAG Agent CLI"""
    pass


@cli.command()
@click.option('--data-path', default=None, help='Path to complaint data file')
@click.option('--force-rebuild', is_flag=True, help='Force rebuild of vector store')
def init(data_path: Optional[str], force_rebuild: bool):
    """Initialize the RAG agent and build vector store"""
    try:
        agent = ComplaintRAGAgent(data_path=data_path)
        agent.initialize(force_rebuild=force_rebuild)
        click.echo(click.style("✓ RAG Agent initialized successfully!", fg='green'))
        
        products = agent.get_available_products()
        if products:
            click.echo(f"\nAvailable products ({len(products)}):")
            for product in products:
                click.echo(f"  - {product}")
    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg='red'), err=True)
        sys.exit(1)


@cli.command()
@click.argument('question')
@click.option('--product', default=None, help='Filter by product')
@click.option('--data-path', default=None, help='Path to complaint data file')
@click.option('-k', default=None, type=int, help='Number of complaints to retrieve')
def ask(question: str, product: Optional[str], data_path: Optional[str], k: Optional[int]):
    """Ask a question about customer complaints"""
    try:
        agent = ComplaintRAGAgent(data_path=data_path)
        agent.initialize()
        
        result = agent.ask(question=question, product=product, k=k)
        
        click.echo(click.style("\n" + "="*80, fg='cyan'))
        click.echo(click.style("ANSWER:", fg='cyan', bold=True))
        click.echo(click.style("="*80, fg='cyan'))
        click.echo(result['answer'])
        
        click.echo(click.style("\n" + "="*80, fg='cyan'))
        click.echo(click.style(f"SOURCES ({result['num_sources']} relevant complaints):", fg='cyan', bold=True))
        click.echo(click.style("="*80, fg='cyan'))
        
        for i, doc in enumerate(result['sources'], 1):
            click.echo(f"\n{i}. Product: {doc.metadata.get('product', 'Unknown')}")
            click.echo(f"   Issue: {doc.metadata.get('issue', 'Unknown')}")
            click.echo(f"   Preview: {doc.page_content[:200]}...")
        
    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg='red'), err=True)
        sys.exit(1)


@cli.command()
@click.argument('question')
@click.argument('products', nargs=-1, required=True)
@click.option('--data-path', default=None, help='Path to complaint data file')
@click.option('-k', default=3, type=int, help='Number of complaints per product')
def compare(question: str, products: tuple, data_path: Optional[str], k: int):
    """Compare complaints across multiple products"""
    try:
        agent = ComplaintRAGAgent(data_path=data_path)
        agent.initialize()
        
        result = agent.compare(
            question=question,
            products=list(products),
            k_per_product=k
        )
        
        click.echo(click.style("\n" + "="*80, fg='cyan'))
        click.echo(click.style("COMPARATIVE ANALYSIS:", fg='cyan', bold=True))
        click.echo(click.style("="*80, fg='cyan'))
        click.echo(f"Products: {', '.join(result['products_compared'])}")
        click.echo()
        click.echo(result['answer'])
        
        click.echo(click.style("\n" + "="*80, fg='cyan'))
        click.echo(click.style(f"SOURCES ({result['num_sources']} total complaints):", fg='cyan', bold=True))
        click.echo(click.style("="*80, fg='cyan'))
        
        for i, doc in enumerate(result['sources'], 1):
            click.echo(f"\n{i}. Product: {doc.metadata.get('product', 'Unknown')}")
            click.echo(f"   Issue: {doc.metadata.get('issue', 'Unknown')}")
            click.echo(f"   Preview: {doc.page_content[:150]}...")
        
    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg='red'), err=True)
        sys.exit(1)


@cli.command()
@click.option('--data-path', default=None, help='Path to complaint data file')
def products(data_path: Optional[str]):
    """List available products in the dataset"""
    try:
        agent = ComplaintRAGAgent(data_path=data_path)
        agent.initialize()
        
        available_products = agent.get_available_products()
        
        click.echo(click.style(f"\nAvailable Products ({len(available_products)}):", fg='cyan', bold=True))
        click.echo(click.style("="*80, fg='cyan'))
        
        for product in sorted(available_products):
            click.echo(f"  • {product}")
        
    except Exception as e:
        click.echo(click.style(f"✗ Error: {str(e)}", fg='red'), err=True)
        sys.exit(1)


@cli.command()
def interactive():
    """Start interactive query mode"""
    click.echo(click.style("Starting Interactive Mode...", fg='cyan', bold=True))
    click.echo("Type 'exit' or 'quit' to exit, 'help' for commands\n")
    
    try:
        agent = ComplaintRAGAgent()
        agent.initialize()
        
        products = agent.get_available_products()
        click.echo(click.style(f"Available products: {', '.join(products[:5])}...", fg='yellow'))
        
        while True:
            try:
                question = click.prompt('\n➤ Your question', type=str)
                
                if question.lower() in ['exit', 'quit']:
                    click.echo(click.style("Goodbye!", fg='cyan'))
                    break
                
                if question.lower() == 'help':
                    click.echo("\nCommands:")
                    click.echo("  - Ask any question about complaints")
                    click.echo("  - Type 'products' to see available products")
                    click.echo("  - Type 'exit' or 'quit' to exit")
                    continue
                
                if question.lower() == 'products':
                    click.echo(f"\nAvailable products: {', '.join(products)}")
                    continue
                
                # Ask question
                result = agent.ask(question)
                
                click.echo(click.style("\n📊 Answer:", fg='green', bold=True))
                click.echo(result['answer'])
                click.echo(click.style(f"\n(Based on {result['num_sources']} relevant complaints)", fg='yellow'))
                
            except KeyboardInterrupt:
                click.echo(click.style("\n\nGoodbye!", fg='cyan'))
                break
            except Exception as e:
                click.echo(click.style(f"\n✗ Error: {str(e)}", fg='red'))
                continue
    
    except Exception as e:
        click.echo(click.style(f"✗ Error initializing: {str(e)}", fg='red'), err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
