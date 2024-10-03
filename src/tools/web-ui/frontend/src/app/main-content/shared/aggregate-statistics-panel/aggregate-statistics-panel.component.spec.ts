import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AggregateStatisticsPanelComponent } from './aggregate-statistics-panel.component';

describe('AggregateStatisticsPanelComponent', () => {
  let component: AggregateStatisticsPanelComponent;
  let fixture: ComponentFixture<AggregateStatisticsPanelComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ AggregateStatisticsPanelComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AggregateStatisticsPanelComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
